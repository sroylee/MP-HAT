package evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.List;

import model.Configure.PredictionMode;
import tool.MathTool;

public class Prediction {
	private String dataPath;
	private String resultPath;
	private String modelMode;
	private String setting;
	private int nTopics;
	private int nPlatforms;
	private int testBatch;
	private PredictionMode predMode;
	private String outputPath;
	// User platform
	private HashMap<String, String> userPlatforms;
	// CTRL model
	private HashMap<String, double[]> userAuthorities;
	private HashMap<String, double[]> userHubs;
	private HashMap<String, double[]> userInterests;
	private HashMap<String, double[][]> userPlatformPreferences;

	// Common-Neighbor
	private HashMap<String, HashMap<Integer, HashSet<String>>> userFollowers;
	private HashMap<String, HashMap<Integer, HashSet<String>>> userFollowees;
	private HashMap<String, Double> traditionalAuthorities;
	// HIST
	private HashMap<String, Double> traditionalHubs;
	// CTR
	private HashMap<String, double[]> userUserLatentFactors;
	private HashMap<String, double[]> userItemLatentFactors;
	// data
	private HashMap<Integer, HashSet<String>> allPositiveLinks;
	private HashMap<String, Integer[]> userTestPositiveLinkCount;
	private HashMap<String, Integer[]> userTestNegativeLinkCount;
	private HashMap<Integer, HashSet<String>> selectedNonLinks;

	private int[] nTests;
	private String[] users;
	private String[] testSrcUsers;
	private String[] testDesUsers;
	private int[] testLabels;
	private int[] testPlatforms;
	private double[] predictionScores;

	/***
	 * read dataset from folder "path"
	 * 
	 * @param dataPath
	 */
	public Prediction(String _path, String _resultPath, int _nTopics,
			int _nPlatforms, int _testBatch, PredictionMode _predMode, String _outputPath) {
		this.dataPath = _path;
		this.resultPath = _resultPath;
		this.nTopics = _nTopics;
		this.nPlatforms = _nPlatforms;
		this.testBatch = _testBatch;
		this.predMode = _predMode;
		this.outputPath = String.format("%s/%d", _outputPath, nTopics);
//		if (predMode == PredictionMode.HAT) {
//			this.outputPath = String.format("%s/%d",_outputPath, nTopics);
//		} else if (predMode == PredictionMode.WTFW) {
//			this.outputPath = String.format("%s/%d", _outputPath, nTopics);
//		} else if (predMode == PredictionMode.CTR) {
//			this.outputPath = String.format("%s/%d", _outputPath, nTopics);
//		} else if (predMode == PredictionMode.COMMON_INTEREST) {
//			this.outputPath = String.format("%s/%d", _outputPath, nTopics);
//		} else {
//			this.outputPath = _outputPath;
//		}

		File theDir = new File(outputPath);

		// if the directory does not exist, create it
		if (!theDir.exists()) {
			System.out.println("creating directory: " + theDir.getAbsolutePath());
			try {
				theDir.mkdirs();
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(-1);
			}
		}
	}

	public void evaluate() {
		System.out.println("loading testing data");
		String relationshipFile = String.format("%s/relationships.csv", dataPath);
		String userFile = String.format("%s/users.csv", dataPath);
		String hitsFile = String.format("%s/user_hits.csv", dataPath);
		String wtfwFile = String.format("%s/wtfw_results.csv", dataPath);
		// int neighhorSize = loadUserNeighbors(relationshipFile);
		int neighhorSize = loadUserDirectedNeighbors(relationshipFile);
		System.out.println("loaded neighbors of " + neighhorSize + " users");
		// loadTestData(relationshipFile, userFile);

		// output_NonLinks();

		if (predMode == PredictionMode.MPHAT) {
			loadTestData(relationshipFile, userFile);

			String authFilePath = String.format("%s/l_userAuthorityDistributions.csv", resultPath);
			int authSize = loadUserAuthorities(authFilePath, nTopics);
			System.out.println("loaded authorities of " + authSize + " users");

			String hubFilePath = String.format("%s/l_userHubDistributions.csv", resultPath);
			int hubSize = loadUserHubs(hubFilePath, nTopics);
			System.out.println("loaded hubs of " + hubSize + " users");

			String preferenceFilePath = String.format("%s/l_userTopicalPlatformPreferenceDistributions.csv",
					resultPath);
			int preferenceSize = loadUserPlatformPreferences(preferenceFilePath, nTopics, nPlatforms);
			System.out.println("loaded platform of " + preferenceSize + " users");

			System.out.println("compute prediction scores");
			computeMPHATScores();

		} else if (predMode == PredictionMode.HAT) {
			loadTestData(relationshipFile, userFile);
			String authFilePath = String.format("%s/l_userAuthorityDistributions.csv", resultPath);
			int authSize = loadUserAuthorities(authFilePath, nTopics);
			System.out.println("loaded authorities of " + authSize + " users");

			String hubFilePath = String.format("%s/l_userHubDistributions.csv", resultPath);
			int hubSize = loadUserHubs(hubFilePath, nTopics);
			System.out.println("loaded hubs of " + hubSize + " users");

			System.out.println("compute prediction scores");
			computeHATScores();

		} else if (predMode == PredictionMode.CTR) {
			loadTestData(relationshipFile, userFile);
			loadCTR();
			computeCTRScores();
		}

		else if (predMode == PredictionMode.COMMON_INTEREST) {
			loadTestData(relationshipFile, userFile);

			String interestFilePath = String.format("%s/l_GibbUserTopicalInterestDistributions.csv",
					resultPath);
			int interestSize = loadUserInterests(interestFilePath, nTopics);
			System.out.println("loaded interests of " + interestSize + " users");

			System.out.println("compute prediction scores");
			computeCommonInterestScores();
		} else if (predMode == PredictionMode.COMMON_NEIGHBOR) {
			loadTestData(relationshipFile, userFile);

			// computeCommonNeighborScores();
			computeCommonDirectedNeighborScores();
		} else if (predMode == PredictionMode.HITS) {
			loadTestData(relationshipFile, userFile);

			loadTraditionalHITS(hitsFile);
			computeHITSScores();
		} else if (predMode == PredictionMode.WTFW) {
			loadTestData(relationshipFile, userFile);

			loadWTFWScores(wtfwFile);
		}

		output_PredictionScores();
		output_EvaluatePlatformSpecificUserLevelPrecisionRecall(5);

	}

	private int loadUserAuthorities(String filename, int nTopics) {
		BufferedReader br = null;
		String line = null;
		double[] authorities;
		userAuthorities = new HashMap<String, double[]>();
		try {
			File authFile = new File(filename);
			br = new BufferedReader(new FileReader(authFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				authorities = new double[nTopics];
				for (int i = 0; i < nTopics; i++) {
					authorities[i] = Double.parseDouble(tokens[i + 1]);
				}
				userAuthorities.put(uid, authorities);
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
		return userAuthorities.size();
	}

	private int loadUserHubs(String filename, int nTopics) {
		BufferedReader br = null;
		String line = null;
		double[] hubs;
		userHubs = new HashMap<String, double[]>();
		try {
			File hubFile = new File(filename);
			br = new BufferedReader(new FileReader(hubFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				hubs = new double[nTopics];
				for (int i = 0; i < nTopics; i++) {
					hubs[i] = Double.parseDouble(tokens[i + 1]);
				}
				userHubs.put(uid, hubs);
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
		return userHubs.size();
	}

	private int loadUserInterests(String filename, int nTopics) {
		BufferedReader br = null;
		String line = null;
		double[] interests;
		userInterests = new HashMap<String, double[]>();
		try {
			File interestFile = new File(filename);
			br = new BufferedReader(new FileReader(interestFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				interests = new double[nTopics];
				for (int i = 0; i < nTopics; i++) {
					interests[i] = Double.parseDouble(tokens[i + 1]);
				}
				userInterests.put(uid, interests);
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
		return userInterests.size();
	}

	private int loadUserPlatformPreferences(String filename, int nTopics, int nPlatforms) {
		BufferedReader br = null;
		String line = null;
		double[][] preferences;
		userPlatformPreferences = new HashMap<String, double[][]>();

		// initialize values
		for (int i = 0; i < users.length; i++) {
			preferences = new double[nTopics][nPlatforms];
			for (int z = 0; z < nTopics; z++) {
				for (int p = 0; p < nPlatforms; p++) {
					preferences[z][p] = Double.NEGATIVE_INFINITY;
				}
			}
			userPlatformPreferences.put(users[i], preferences);
		}

		try {
			File interestFile = new File(filename);
			br = new BufferedReader(new FileReader(interestFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				preferences = userPlatformPreferences.get(uid);
				int topic = Integer.parseInt(tokens[1]);
				for (int p = 0; p < nPlatforms; p++) {
					preferences[topic][p] = Double.parseDouble(tokens[p + 2]);
				}
				// userPlatformPreferences.put(uid, preferences); no need to put
				// again
			}
			br.close();

			for (int u = 0; u < users.length; u++) {
				preferences = userPlatformPreferences.get(users[u]);
				for (int z = 0; z < nTopics; z++) {
					preferences[z] = MathTool.softmax(preferences[z]);
				}
			}

		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
		return userPlatformPreferences.size();
	}

	private int loadUserDirectedNeighbors(String relationshipFile) {
		BufferedReader br = null;
		String line = null;
		userFollowers = new HashMap<String, HashMap<Integer, HashSet<String>>>();
		userFollowees = new HashMap<String, HashMap<Integer, HashSet<String>>>();
		try {
			File linkFile = new File(relationshipFile);

			br = new BufferedReader(new FileReader(linkFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				String vid = tokens[1];
				int platform = Integer.parseInt(tokens[2]);
				int flag = Integer.parseInt(tokens[3]);
				if (flag == 1) {
					if (userFollowees.containsKey(uid)) {
						HashMap<Integer, HashSet<String>> platforms = userFollowees.get(uid);
						if (platforms.containsKey(platform)) {
							platforms.get(platform).add(vid);
						} else {
							HashSet<String> followees = new HashSet<String>();
							followees.add(vid);
							platforms.put(platform, followees);
						}
					} else {
						HashSet<String> followees = new HashSet<String>();
						followees.add(vid);
						HashMap<Integer, HashSet<String>> platforms = new HashMap<Integer, HashSet<String>>();
						platforms.put(platform, followees);
						userFollowees.put(uid, platforms);
					}
					if (userFollowers.containsKey(vid)) {
						HashMap<Integer, HashSet<String>> platforms = userFollowers.get(vid);
						if (platforms.containsKey(platform)) {
							platforms.get(platform).add(uid);
						} else {
							HashSet<String> followers = new HashSet<String>();
							followers.add(uid);
							platforms.put(platform, followers);
						}
					} else {
						HashSet<String> followers = new HashSet<String>();
						followers.add(uid);
						HashMap<Integer, HashSet<String>> platforms = new HashMap<Integer, HashSet<String>>();
						platforms.put(platform, followers);
						userFollowers.put(vid, platforms);
					}
				}
			}
			br.close();

		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
		return userFollowers.size();
	}

	private void loadTraditionalHITS(String filename) {
		BufferedReader br = null;
		String line = null;
		traditionalAuthorities = new HashMap<String, Double>();
		traditionalHubs = new HashMap<String, Double>();
		try {
			File hitFile = new File(filename);
			br = new BufferedReader(new FileReader(hitFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				double authority = Double.parseDouble(tokens[1]);
				double hub = Double.parseDouble(tokens[2]);
				traditionalAuthorities.put(uid, authority);
				traditionalHubs.put(uid, hub);
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void loadCTR() {
		try {
			userUserLatentFactors = new HashMap<String, double[]>();
			HashMap<Integer, String> userIndex2Id = new HashMap<Integer, String>();
			String filename = String.format("%s/ctr/user_index_id.txt", dataPath);
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = null;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				userIndex2Id.put(Integer.parseInt(tokens[0]), tokens[1]);
			}
			br.close();

			filename = String.format("%s/final-U.dat", resultPath);
			int uIndex = 0;
			br = new BufferedReader(new FileReader(filename));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(" ");
				double[] factors = new double[nTopics];
				for (int i = 0; i < nTopics; i++) {
					factors[i] = Double.parseDouble(tokens[i]);
				}
				userUserLatentFactors.put(userIndex2Id.get(uIndex), factors);
				uIndex++;
			}
			br.close();

			userItemLatentFactors = new HashMap<String, double[]>();
			userIndex2Id = new HashMap<Integer, String>();
			filename = String.format("%s/ctr/item_index_id.txt", dataPath);
			br = new BufferedReader(new FileReader(filename));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				userIndex2Id.put(Integer.parseInt(tokens[0]), tokens[1]);
			}
			br.close();

			filename = String.format("%s/final-V.dat", resultPath);
			uIndex = 0;
			br = new BufferedReader(new FileReader(filename));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(" ");
				double[] factors = new double[nTopics];
				for (int i = 0; i < nTopics; i++) {
					factors[i] = Double.parseDouble(tokens[i]);
				}
				userItemLatentFactors.put(userIndex2Id.get(uIndex), factors);
				uIndex++;
			}
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void loadWTFWScores(String filename) {
		BufferedReader br = null;
		String line = null;
		int index = 0;
		try {
			File wtfwFile = new File(filename);
			br = new BufferedReader(new FileReader(wtfwFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				double score = Double.parseDouble(tokens[2]);
				predictionScores[index] = score;
				index++;
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void loadTestData(String _relationshipFile, String _userFile) {
		BufferedReader br = null;
		String line = null;
		int nUser = 0;
		nTests = new int[nPlatforms];
		for (int p = 0; p < nPlatforms; p++) {
			nTests[p] = 0;
		}
		userPlatforms = new HashMap<String, String>();
		userTestPositiveLinkCount = new HashMap<String, Integer[]>();
		userTestNegativeLinkCount = new HashMap<String, Integer[]>();
		allPositiveLinks = new HashMap<Integer, HashSet<String>>();
		selectedNonLinks = new HashMap<Integer, HashSet<String>>();

		try {
			File userFile = new File(_userFile);
			br = new BufferedReader(new FileReader(userFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				nUser++;
			}
			br.close();

			users = new String[nUser];

			int iUser = 0;
			br = new BufferedReader(new FileReader(userFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				users[iUser] = tokens[0];
				String platforms = "";
				for (int p = 0; p < nPlatforms; p++) {
					platforms = platforms + tokens[2 + p] + " ";
				}
				platforms = platforms.trim();
				userPlatforms.put(users[iUser], platforms);
				iUser++;
			}
			br.close();

			File linkFile = new File(_relationshipFile);
			br = new BufferedReader(new FileReader(linkFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				String vid = tokens[1];
				int platform = Integer.parseInt(tokens[2]);
				String link = uid.trim() + " " + vid.trim();
				if (allPositiveLinks.containsKey(platform)) {
					allPositiveLinks.get(platform).add(link);
				} else {
					HashSet<String> links = new HashSet<String>();
					links.add(link);
					allPositiveLinks.put(platform, links);
				}
				int batch = Integer.parseInt(tokens[3]);
				if (batch == testBatch) {
					nTests[platform]++;
					if (userTestPositiveLinkCount.containsKey(uid)) {
						userTestPositiveLinkCount.get(uid)[platform]++;
					} else {
						Integer[] counts = new Integer[nPlatforms];
						for (int p = 0; p < nPlatforms; p++) {
							counts[p] = 0;
						}
						counts[platform] = 1;
						userTestPositiveLinkCount.put(uid, counts);
					}
				}
			}
			br.close();

			// find non-links with common neighbor (2-hops)

			for (int p = 0; p < nPlatforms; p++) {
				selectedNonLinks.put(p, new HashSet<String>());
				for (int u = 0; u < users.length; u++) {

					System.out.printf("selecting non-links: platform = %d, u = %d/%d\n", p, u, users.length);

					String uid = users[u];
					if (userFollowers.containsKey(uid) == false) {
						continue;
					}
					if (userFollowers.get(uid).containsKey(p) == false) {
						continue;
					}
					HashSet<String> uSelectedNonFollowees = new HashSet<String>();
					HashSet<String> uFollowers = userFollowers.get(uid).get(p);
					for (String wid : uFollowers) {
						if (userFollowers.containsKey(wid) == false) {
							continue;
						}
						if (userFollowers.get(wid).containsKey(p) == false) {
							continue;
						}
						HashSet<String> wFollowers = userFollowers.get(wid).get(p);
						for (String vid : wFollowers) {
							String nonLink = uid.trim() + " " + vid.trim();
							if (allPositiveLinks.get(p).contains(nonLink)) {
								continue;
							}
							uSelectedNonFollowees.add(vid);
						}
					}
					for (String vid : uSelectedNonFollowees) {
						String nonLink = uid.trim() + " " + vid.trim();
						selectedNonLinks.get(p).add(nonLink);
						nTests[p]++;
						if (userTestNegativeLinkCount.containsKey(uid)) {
							userTestNegativeLinkCount.get(uid)[p]++;
						} else {
							Integer[] counts = new Integer[nPlatforms];
							for (int pp = 0; pp < nPlatforms; pp++) {
								counts[pp] = 0;
							}
							counts[p] = 1;
							userTestNegativeLinkCount.put(uid, counts);
						}
					}
				}
			}

			int nAllTests = 0;
			for (int p = 0; p < nPlatforms; p++) {
				nAllTests += nTests[p];
			}
			testSrcUsers = new String[nAllTests];
			testDesUsers = new String[nAllTests];
			testLabels = new int[nAllTests];
			testPlatforms = new int[nAllTests];
			predictionScores = new double[nAllTests];

			int iTest = 0;
			for (Map.Entry<Integer, HashSet<String>> pair : selectedNonLinks.entrySet()) {
				for (String nonlink : pair.getValue()) {
					String[] nonLinkPair = nonlink.split(" ");
					testSrcUsers[iTest] = nonLinkPair[0];
					testDesUsers[iTest] = nonLinkPair[1];
					testLabels[iTest] = 0;
					testPlatforms[iTest] = pair.getKey();
					iTest++;
				}
			}

			System.out.println("Generated " + iTest + " non links");

			br = new BufferedReader(new FileReader(linkFile.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String uid = tokens[0];
				String vid = tokens[1];
				int platform = Integer.parseInt(tokens[2]);
				int batch = Integer.parseInt(tokens[3]);
				if (batch == testBatch) {
					testSrcUsers[iTest] = uid;
					testDesUsers[iTest] = vid;
					testLabels[iTest] = 1;
					testPlatforms[iTest] = platform;
					iTest++;
				}
			}
			br.close();

			System.out.println("Loaded " + iTest + " links");

		} catch (

		Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void computeMPHATScores() {
		String uid = "";
		String vid = "";
		int platform = 0;
		double[] Hu;
		double[] Av;
		double[][] Eta_u;
		double[][] Eta_v;
		double HupAvp = 0;
		for (int i = 0; i < testLabels.length; i++) {
			uid = testSrcUsers[i];
			vid = testDesUsers[i];
			platform = testPlatforms[i];
			Hu = userHubs.get(uid);
			Av = userAuthorities.get(vid);
			Eta_u = userPlatformPreferences.get(uid);
			Eta_v = userPlatformPreferences.get(vid);
			HupAvp = 0;
			for (int z = 0; z < nTopics; z++) {
				HupAvp += Hu[z] * Eta_u[z][platform] * Av[z] * Eta_v[z][platform];
			}
			predictionScores[i] = HupAvp;
		}
	}

	private void computeHATScores() {
		String uid = "";
		String vid = "";
		double[] Hu;
		double[] Av;
		double HuAv = 0;
		for (int i = 0; i < testLabels.length; i++) {
			uid = testSrcUsers[i];
			vid = testDesUsers[i];
			Hu = userHubs.get(uid);
			Av = userAuthorities.get(vid);
			HuAv = 0;
			for (int z = 0; z < Hu.length; z++) {
				HuAv += Hu[z] * Av[z];
			}
			predictionScores[i] = HuAv;

		}
	}

	private void computeCTRScores() {
		String uid = null;
		String vid = null;
		double[] uU;
		double[] vV;
		double dotProduct = 0;
		for (int i = 0; i < testLabels.length; i++) {
			uid = testSrcUsers[i];
			vid = testDesUsers[i];
			uU = userUserLatentFactors.get(uid);
			vV = userItemLatentFactors.get(vid);

			// System.out.printf("u = %s v = %s isNull(uU) = %s isNull(vV) =
			// %s\n", uid, vid, (uU == null), (vV == null));

			dotProduct = 0;
			for (int z = 0; z < nTopics; z++) {
				dotProduct += uU[z] * vV[z];
			}
			predictionScores[i] = dotProduct;
		}
	}

	private void computeCommonInterestScores() {
		String uid = "";
		String vid = "";
		double[] Iu;
		double[] Iv;
		double IuIv = 0;
		for (int i = 0; i < testLabels.length; i++) {
			uid = testSrcUsers[i];
			vid = testDesUsers[i];
			Iu = userInterests.get(uid);
			Iv = userInterests.get(vid);
			IuIv = 0;
			for (int z = 0; z < Iu.length; z++) {
				IuIv += Iu[z] * Iv[z];
			}
			predictionScores[i] = IuIv;
		}
	}

	private void computeCommonDirectedNeighborScores() {
		String uid = "";
		String vid = "";
		for (int i = 0; i < testLabels.length; i++) {
			uid = testSrcUsers[i];
			vid = testDesUsers[i];
			if (userFollowees.containsKey(uid) == false || userFollowers.containsKey(vid) == false) {
				predictionScores[i] = 0f;
				continue;
			}
			int platform = testPlatforms[i];
			if (userFollowees.get(uid).containsKey(platform) == false
					|| userFollowers.get(vid).containsKey(platform) == false) {
				predictionScores[i] = 0f;
				continue;
			}

			HashSet<String> uNeighborsSet = userFollowees.get(uid).get(platform);
			HashSet<String> vNeighborsSet = userFollowers.get(vid).get(platform);

			HashSet<String> unionSet = new HashSet<String>();
			unionSet.addAll(uNeighborsSet);
			unionSet.addAll(vNeighborsSet);

			HashSet<String> intersectionSet = new HashSet<String>();
			intersectionSet.addAll(uNeighborsSet);
			intersectionSet.retainAll(vNeighborsSet);

			predictionScores[i] = (float) intersectionSet.size() / (float) unionSet.size();
		}
	}

	private void computeHITSScores() {
		String uid = "";
		String vid = "";
		double HuAv = 0;
		double Hu = 0f;
		double Av = 0f;
		for (int i = 0; i < testLabels.length; i++) {
			uid = testSrcUsers[i];
			vid = testDesUsers[i];
			Hu = traditionalHubs.get(uid);
			Av = traditionalAuthorities.get(vid);
			HuAv = Hu * Av;
			predictionScores[i] = HuAv;
		}
	}

	private void output_PredictionScores() {
		try {
			File f = new File(outputPath + "/" + predMode + "_pred_scores.csv");
			FileWriter fo = new FileWriter(f, false);

			for (int i = 0; i < testLabels.length; i++) {
				fo.write(testSrcUsers[i] + "," + testDesUsers[i] + "," + testLabels[i] + "," + predictionScores[i]
						+ "\n");
			}
			fo.close();
		} catch (Exception e) {
			System.out.println("Error in writing out post topic top words to file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void output_EvaluatePlatformSpecificUserLevelPrecisionRecall(int k) {
		double[] precision = new double[k];
		double[] recall = new double[k];

		HashMap<String, HashMap<Integer, ArrayList<Integer>>> UserLinkLabels = new HashMap<String, HashMap<Integer, ArrayList<Integer>>>();
		for (int i = 0; i < testSrcUsers.length; i++) {
			String uid = testSrcUsers[i];
			int platform = testPlatforms[i];
			if (UserLinkLabels.containsKey(uid)) {
				if (UserLinkLabels.get(uid).containsKey(platform)) {
					continue;
				}
				UserLinkLabels.get(uid).put(platform, new ArrayList<Integer>());
			} else {
				HashMap<Integer, ArrayList<Integer>> platforms = new HashMap<Integer, ArrayList<Integer>>();
				platforms.put(platform, new ArrayList<Integer>());
				UserLinkLabels.put(uid, platforms);
			}
		}

		Map<Integer, Double> mapPredictionScores = new HashMap<Integer, Double>();
		for (int s = 0; s < predictionScores.length; s++) {
			mapPredictionScores.put(s, predictionScores[s]);
		}
		List<Entry<Integer, Double>> sortedScores = sortByValue(mapPredictionScores);
		for (Map.Entry<Integer, Double> entry : sortedScores) {
			int index = entry.getKey();
			String uid = testSrcUsers[index];
			int platform = testPlatforms[index];
			UserLinkLabels.get(uid).get(platform).add(testLabels[index]);
		}

		for (int p = 0; p < nPlatforms; p++) {
			for (int i = 0; i < k; i++) {
				int checkPosCount = 0;
				int currK = i + 1;
				float sumPrecision = 0;
				float sumRecall = 0;
				int count = 0;
				for (int u = 0; u < users.length; u++) {
					String uid = users[u];
					String[] uPlatform = userPlatforms.get(uid).split(" ");
					if (uPlatform[p].equals("1")) {
						int posCount = 0;
						if (userTestPositiveLinkCount.containsKey(uid)
								&& userTestPositiveLinkCount.get(uid)[p] >= currK) {
							if (userTestNegativeLinkCount.containsKey(uid)
									&& userTestNegativeLinkCount.get(uid)[p] >= currK) {
								checkPosCount += userTestPositiveLinkCount.get(uid)[p];
								count++;
								ArrayList<Integer> labels = UserLinkLabels.get(uid).get(p);
								for (int j = 0; j < currK; j++) {
									if (labels.get(j) == 1) {
										posCount++;
									}
								}
								sumPrecision += (float) posCount / (float) currK;
								sumRecall += (float) posCount / (float) userTestPositiveLinkCount.get(uid)[p];
							}
						}
					}

				}
				System.out.println("#PositiveLinks@" + k + ": " + checkPosCount);
				precision[i] = sumPrecision / count;
				recall[i] = sumRecall / count;

			}

			int[] rank = new int[users.length];
			int iRank = 0;
			for (int u = 0; u < users.length; u++) {
				String uid = users[u];
				String[] uPlatform = userPlatforms.get(uid).split(" ");
				if (uPlatform[p].equals("1")) {
					rank[iRank] = 0;
					int posCount = 0;
					if (userTestPositiveLinkCount.containsKey(uid) && userTestNegativeLinkCount.containsKey(uid)) {
						ArrayList<Integer> labels = UserLinkLabels.get(uid).get(p);
						for (int j = 0; j < labels.size(); j++) {
							if (labels.get(j) == 1) {
								posCount++;
								if (posCount == 1) {
									rank[iRank] = j + 1;
									break;
								}
							}
						}
					}
					iRank++;
				}
			}

			double sumMRR = 0f;
			double mrr = 0f;
			int countMRR = 0;
			for (int i = 0; i < rank.length; i++) {
				if (rank[i] != 0) {
					sumMRR += (double) 1 / (double) rank[i];
					countMRR++;
				}
			}
			mrr = sumMRR / countMRR;

			try {
				File f = new File(outputPath + "/" + p + "_" + predMode + "_UserLevel_PrecisionRecall.csv");
				FileWriter fo = new FileWriter(f, false);

				for (int i = 0; i < k; i++) {
					fo.write(i + "," + precision[i] + "," + recall[i] + "\n");
				}
				fo.write("MRR," + mrr + "," + mrr + "\n");
				fo.close();
				fo.close();
			} catch (Exception e) {
				System.out.println("Error in writing out post topic top words to file!");
				e.printStackTrace();
				System.exit(0);
			}
		}
	}

	private <K, V extends Comparable<? super V>> List<Entry<K, V>> sortByValue(Map<K, V> map) {
		List<Entry<K, V>> sortedEntries = new ArrayList<Entry<K, V>>(map.entrySet());
		Collections.sort(sortedEntries, new Comparator<Entry<K, V>>() {
			@Override
			public int compare(Entry<K, V> e1, Entry<K, V> e2) {
				return e2.getValue().compareTo(e1.getValue());
			}
		});

		return sortedEntries;
	}

	public void output_NonLinks() {
		try {
			File f = new File(outputPath + "/" + "l_predictionTestLinks.csv");
			FileWriter fo = new FileWriter(f, false);

			for (int i = 0; i < testLabels.length; i++) {
				fo.write(testSrcUsers[i] + "," + testDesUsers[i] + "," + testLabels[i] + "\n");
			}
			fo.close();
		} catch (Exception e) {
			System.out.println("Error in writing out post topic top words to file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

}

package evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import tool.HungaryMethod;
import tool.MathTool;
import tool.Vector;

public class CompareWithGroundTruth {

	private boolean checkTopic = true;
	private boolean userInterest = true;
	private boolean userAuthority = true;
	private boolean userHub = true;
	private boolean userPlatformPreference = false;

	private String groundtruthPath;
	private String learntPath;
	private String distance;
	private String outputPath;

	private int nTopics;
	private int nUsers;
	private int nPlatforms =2;
	private int nWords;

	public HashMap<String, Integer> userId2Index;
	public HashMap<Integer, String> userIndex2Id;

	// groundtruth params are prefixed by "g"
	private double[][] g_topicWordDistributions;
	private double[][] g_userTopicInterestDistributions;
	private double[][][] g_userPlatformPreferenceDistributions;
	private double[][] g_userAuthorityDistributions;
	private double[][] g_userHubDistributions;

	// learnt params are prefixed by "l"
	private double[][] l_topicWordDistributions;
	private double[][] l_userTopicInterestDistributions;
	private double[][][] l_userPlatformPreferenceDistributions;
	private double[][] l_userAuthorityDistributions;
	private double[][] l_userHubDistributions;

	private int[] glMatch;
	private int[] lgMatch;
	private double[][] topicDistance;

	public CompareWithGroundTruth(String _groundtruthPath, String _learntPath, String _distance, String _outputPath) {
		groundtruthPath = _groundtruthPath;
		learntPath = _learntPath;
		distance = _distance;
		outputPath = _outputPath;
	}

	private void getGroundTruth() {
		try {
			String filename;
			BufferedReader br;
			String line = null;
			// Topics Words Distributions
			if (checkTopic) {
				filename = String.format("%s/topicWordDistributions.csv", groundtruthPath);
				br = new BufferedReader(new FileReader(filename));
				nTopics = 1;
				line = br.readLine();
				nWords = line.split(",").length - 1;
				while ((line = br.readLine()) != null) {
					nTopics++;
				}
				br.close();

				g_topicWordDistributions = new double[nTopics][nWords];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int t = Integer.parseInt(tokens[0]);
					for (int i = 1; i < tokens.length; i++) {
						g_topicWordDistributions[t][i - 1] = Double.parseDouble(tokens[i]);
					}
				}
				br.close();
			}

			// User Topic Interest Distributions
			if (userInterest) {
				filename = String.format("%s/userLatentFactors.csv", groundtruthPath);
				br = new BufferedReader(new FileReader(filename));
				nUsers = 0;
				line = null;
				while ((line = br.readLine()) != null) {
					nUsers++;
				}
				br.close();
				userId2Index = new HashMap<String, Integer>();
				g_userTopicInterestDistributions = new double[nUsers][nTopics];
				br = new BufferedReader(new FileReader(filename));
				int u = 0;
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					userId2Index.put(tokens[0], u);
					for (int t = 1; t < tokens.length; t++) {
						g_userTopicInterestDistributions[u][t - 1] = Double.parseDouble(tokens[t]);
					}
					u++;
				}
				br.close();
			}

			// User Authority Distributions
			if (userAuthority) {
				filename = String.format("%s/userAuthorities.csv", groundtruthPath);
				g_userAuthorityDistributions = new double[nUsers][nTopics];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int u = userId2Index.get(tokens[0]);
					for (int t = 1; t < tokens.length; t++) {
						g_userAuthorityDistributions[u][t - 1] = Double.parseDouble(tokens[t]);
					}
				}
				br.close();
			}

			// User Hub Distributions
			if (userHub) {
				filename = String.format("%s/userHubs.csv", groundtruthPath);
				g_userHubDistributions = new double[nUsers][nTopics];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int u = userId2Index.get(tokens[0]);
					for (int t = 1; t < tokens.length; t++) {
						g_userHubDistributions[u][t - 1] = Double.parseDouble(tokens[t]);
					}
				}
				br.close();
			}

			// User Platform Preference
			if (userPlatformPreference) {
				filename = String.format("%s/userPlatformPreference.csv", groundtruthPath);
				g_userPlatformPreferenceDistributions = new double[nUsers][nTopics][nPlatforms];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int u = userId2Index.get(tokens[0]);
					int z = Integer.parseInt(tokens[1]);
					for (int p = 2; p < tokens.length; p++) {
						g_userPlatformPreferenceDistributions[u][z][p-2] = Double.parseDouble(tokens[p]);
					}
				}
				br.close();
			}				
			

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void getLearntParams() {
		try {
			// Topics Words Distributions
			String filename;
			BufferedReader br;
			String line = null;

			if (checkTopic) {
				filename = String.format("%s/l_topicalWordDistributions.csv", learntPath);
				br = new BufferedReader(new FileReader(filename));
				nTopics = 1;
				line = br.readLine();
				nWords = line.split(",").length - 1;
				while ((line = br.readLine()) != null) {
					nTopics++;
				}
				br.close();

				l_topicWordDistributions = new double[nTopics][nWords];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int t = Integer.parseInt(tokens[0]);
					for (int i = 1; i < tokens.length; i++) {
						l_topicWordDistributions[t][i - 1] = Double.parseDouble(tokens[i]);
					}
				}
				br.close();
			}

			// User Topic Interest Distributions
			if (userInterest) {
				filename = String.format("%s/l_userTopicalInterestDistributions.csv", learntPath);
				// br = new BufferedReader(new FileReader(filename));
				l_userTopicInterestDistributions = new double[nUsers][nTopics];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				int u = 0;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					u = userId2Index.get(tokens[0]);
					for (int t = 1; t < tokens.length; t++) {
						l_userTopicInterestDistributions[u][t - 1] = Double.parseDouble(tokens[t]);
					}
				}
				br.close();
			}

			// User Authority Distributions
			if (userAuthority) {
				filename = String.format("%s/l_userAuthorityDistributions.csv", learntPath);
				l_userAuthorityDistributions = new double[nUsers][nTopics];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int u = userId2Index.get(tokens[0]);
					for (int t = 1; t < tokens.length; t++) {
						l_userAuthorityDistributions[u][t - 1] = Double.parseDouble(tokens[t]);
					}
				}
				br.close();
			}

			// User Hub Distributions
			if (userHub) {
				filename = String.format("%s/l_userHubDistributions.csv", learntPath);
				l_userHubDistributions = new double[nUsers][nTopics];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int u = userId2Index.get(tokens[0]);
					for (int t = 1; t < tokens.length; t++) {
						l_userHubDistributions[u][t - 1] = Double.parseDouble(tokens[t]);
					}
				}
				br.close();
			}

			// User Platform Preference
			if (userPlatformPreference) {
				filename = String.format("%s/l_userTopicalPlatformPreferenceDistributions.csv", learntPath);
				l_userPlatformPreferenceDistributions = new double[nUsers][nTopics][nPlatforms];
				br = new BufferedReader(new FileReader(filename));
				line = null;
				while ((line = br.readLine()) != null) {
					String[] tokens = line.split(",");
					int u = userId2Index.get(tokens[0]);
					int z = Integer.parseInt(tokens[1]);
					for (int p = 2; p < tokens.length; p++) {
						l_userPlatformPreferenceDistributions[u][z][p-2] = Double.parseDouble(tokens[p]);
					}
				}
				br.close();
			}
			
			
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void topicMatching() {

		Vector vector = new Vector();

		topicDistance = new double[nTopics][nTopics];
		for (int t = 0; t < nTopics; t++) {
			for (int k = 0; k < nTopics; k++) {
				if (distance.equals("euclidean")) {
					topicDistance[t][k] = vector.euclideanDistance(g_topicWordDistributions[t],
							l_topicWordDistributions[k]);
				} else {
					topicDistance[t][k] = vector.jensenShallonDistance(g_topicWordDistributions[t],
							l_topicWordDistributions[k]);
				}
				if (topicDistance[t][k] < 0) {
					System.out.println("something wrong!!!!");
					System.exit(-1);
				}
			}
		}
		System.out.println("Cost:");
		for (int t = 0; t < nTopics; t++) {
			System.out.printf("%f", topicDistance[t][0]);
			for (int k = 1; k < nTopics; k++) {
				System.out.printf(" %f", topicDistance[t][k]);
			}
			System.out.println("");
		}

		HungaryMethod matcher = new HungaryMethod(topicDistance);
		glMatch = matcher.execute();
		lgMatch = new int[nTopics];
		for (int i = 0; i < nTopics; i++) {
			int j = glMatch[i];
			lgMatch[j] = i;
		}

		System.out.print("glMatch[]: ");
		for (int i = 0; i < glMatch.length; i++) {
			System.out.print(glMatch[i] + " ");
		}
		System.out.println("");
		System.out.print("lgMatch[]: ");
		for (int i = 0; i < lgMatch.length; i++) {
			System.out.print(lgMatch[i] + " ");
		}
		System.out.println("");

	}

	public void measureGoodness() {
		try {
			System.out.println("getting groundtruth");
			getGroundTruth();
			System.out.println("getting learnt parameters");
			getLearntParams();

			System.out.printf("#words = %d #users = %d #topics = %d\n", nWords, nUsers, nTopics);

			System.out.println("matching topics");
			topicMatching();

			String filename = String.format("%s/topicDistance.csv", outputPath);
			BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
			for (int t = 0; t < nTopics; t++) {
				bw.write(String.format("%d,%f\n", t, topicDistance[t][glMatch[t]]));
			}
			bw.close();

			System.out.println("measuring user topic interest distribution distance");
			Vector vector = new Vector();
			filename = String.format("%s/userTopicInterestDistance.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			Iterator<Map.Entry<String, Integer>> iter = userId2Index.entrySet().iterator();
			while (iter.hasNext()) {
				Map.Entry<String, Integer> pair = iter.next();
				int u = pair.getValue();
				if (distance.equals("euclidean")) {
					bw.write(String.format("%s,%f\n", pair.getKey(),
							vector.weightedEuclideanDistance(MathTool.softmax(g_userTopicInterestDistributions[u]),
									MathTool.softmax(l_userTopicInterestDistributions[u]), glMatch,
									MathTool.softmax(g_userTopicInterestDistributions[u]))));
				} else {
					bw.write(String.format("%s,%f\n", pair.getKey(),
							vector.jensenShallonDistance(g_userTopicInterestDistributions[u],
									l_userTopicInterestDistributions[u], glMatch, lgMatch)));
				}
			}
			bw.close();

			//System.exit(-1);

			System.out.println("measuring user authority distribution distance");
			vector = new Vector();
			filename = String.format("%s/userAuthorityDistance.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			iter = userId2Index.entrySet().iterator();
			int l_topic_max = 0;
			int g_topic_max = 0;
			double l_authority_max = 0.0;
			double g_authority_max = 0.0;
			while (iter.hasNext()) {
				Map.Entry<String, Integer> pair = iter.next();
				int u = pair.getValue();
				l_authority_max = 0.0;
				g_authority_max = 0.0;
				for (int k = 0; k < nTopics; k++) {
					if (g_userAuthorityDistributions[u][k] > g_authority_max) {
						g_authority_max = g_userAuthorityDistributions[u][k];
						g_topic_max = k;
					}
				}
				for (int k = 0; k < nTopics; k++) {
					if (l_userAuthorityDistributions[u][k] > l_authority_max) {
						l_authority_max = l_userAuthorityDistributions[u][k];
						l_topic_max = k;
					}
				}
				if (g_topic_max == lgMatch[l_topic_max]) {
					bw.write(pair.getKey() + "," + 1 + "\n");
					// bw.write(String.format("%s,%f\n", pair.getKey(), 1));
				} else {
					bw.write(pair.getKey() + "," + 0 + "\n");
					// bw.write(String.format("%s,%f\n", pair.getKey(), 0));
				}
			}
			bw.close();
			
			System.out.println("measuring user hub distribution distance");
			vector = new Vector();
			filename = String.format("%s/userHubDistance.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			iter = userId2Index.entrySet().iterator();
			l_topic_max = 0;
			g_topic_max = 0;
			while (iter.hasNext()) {
				Map.Entry<String, Integer> pair = iter.next();
				int u = pair.getValue();
				l_authority_max = 0.0;
				g_authority_max = 0.0;
				for (int k = 0; k < nTopics; k++) {
					if (g_userHubDistributions[u][k] > g_authority_max) {
						g_authority_max = g_userHubDistributions[u][k];
						g_topic_max = k;
					}
				}
				for (int k = 0; k < nTopics; k++) {
					if (l_userHubDistributions[u][k] > l_authority_max) {
						l_authority_max = l_userAuthorityDistributions[u][k];
						l_topic_max = k;
					}
				}
				if (g_topic_max == lgMatch[l_topic_max]) {
					bw.write(pair.getKey() + "," + 1 + "\n");
					// bw.write(String.format("%s,%f\n", pair.getKey(), 1));
				} else {
					bw.write(pair.getKey() + "," + 0 + "\n");
					// bw.write(String.format("%s,%f\n", pair.getKey(), 0));
				}
			}
			bw.close();
			
			System.out.println("measuring user platform preferences distance");
			vector = new Vector();
			filename = String.format("%s/userPlatformDistance.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			iter = userId2Index.entrySet().iterator();
			int l_platform_max[] = new int [nTopics];
			int g_platform_max[] = new int [nTopics];
			double l_preference_max = 0.0;
			double g_preference_max = 0.0;
			//l_topic_max = 0;
			//g_topic_max = 0;
			while (iter.hasNext()) {
				Map.Entry<String, Integer> pair = iter.next();
				int u = pair.getValue();
					
				//l_authority_max = 0.0;
				//g_authority_max = 0.0;
				for (int k = 0; k < nTopics; k++) {
					l_preference_max = 0.0;
					g_preference_max = 0.0;	
					for (int p=0; p< nPlatforms; p++){
						if (g_userPlatformPreferenceDistributions[u][k][p] > g_preference_max) {
							g_preference_max = g_userPlatformPreferenceDistributions[u][k][p];
							g_platform_max[k] = p;
						}
						if (l_userPlatformPreferenceDistributions[u][k][p] > l_preference_max) {
							l_preference_max = l_userPlatformPreferenceDistributions[u][k][p];
							l_platform_max[k] = p;
						}
					}
				}
				
				int nCorrect = 0;
				for (int k=0; k<nTopics; k++){
					if (g_platform_max[k] == l_platform_max[lgMatch[k]]) {
						nCorrect ++;
					}
				}
				double result = (double)nCorrect/nTopics;
				bw.write(pair.getKey() + "," + result + "\n");
				
			}
			bw.close();
			

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}

	}

	public static void main(String[] args) {
		 //CompareWithGroundTruth comparator = new CompareWithGroundTruth("/Users/roylee/Documents/Chardonnay/mp-hat/syn_data/groundtruth",
		 //"/Users/roylee/Documents/Chardonnay/mp-hat/syn_data/10", "euclidean",
		 //"/Users/roylee/Documents/Chardonnay/mp-hat/syn_data/evaluation");
		 
		 CompareWithGroundTruth comparator = new CompareWithGroundTruth("E:/users/roylee.2013/MP-HAT/mp-hat/syn_data/groundtruth",
				 "E:/users/roylee.2013/MP-HAT/mp-hat/syn_data/10", "euclidean",
				 "E:/users/roylee.2013/MP-HAT/mp-hat/syn_data/evaluation");

		//CompareWithGroundTruth comparator = new CompareWithGroundTruth("E:/code/java/MP-HAT/mp-hat/output/syn_data",
		//		"E:/code/java/MP-HAT/mp-hat/output/syn_data/10", "euclidean",
		//		"E:/code/java/MP-HAT/mp-hat/output/syn_data/10/evaluation");
		comparator.measureGoodness();
	}

}

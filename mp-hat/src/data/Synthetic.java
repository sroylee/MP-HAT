/***
 * Utilities for synthetic data generation
 */
package data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.GammaDistribution;

import model.Configure.ModelMode;
import tool.MathTool;
import tool.StatTool;

public class Synthetic {

	private class Tuple implements Comparable<Tuple> {
		private int intKey;
		private double doubleValue;

		public Tuple(int _intKey, double _doubleValue) {
			intKey = _intKey;
			doubleValue = _doubleValue;
		}

		public int getIntKey() {
			return intKey;
		}

		public double getDoubleValue() {
			return doubleValue;
		}

		@Override
		public int compareTo(Tuple o) {
			if (doubleValue < o.doubleValue) {
				return -1;
			} else if (doubleValue > o.doubleValue) {
				return 1;
			} else {
				return 0;
			}
		}

	}

	private class Post {
		public int topic;
		public int platform;
		public int[] words;
	}

	private double mass = 0.9;
	private double userSkewness = 0.1;// together with mass, this means, for
										// each user, 90% of her posts are about
										// 10% of topics
	private double topicSkewness = 0.001;// similarly, each topic focuses on
											// 0.1%
											// of words whose probabilities
											// summing
											// up to 99%
	private double singlePlatformProp = 0.0;
	private double platformSkeness;

	private int minNPosts = 100;
	private int maxNPosts = 200;

	private int minNWords = 10;
	private int maxNWords = 20;

	private StatTool statTool = new StatTool();

	public double alpha = 1;
	public double beta = 1;

	public double sigma = 2.0;
	public double delta = 2.0;
	public double omega = 1.0;
	public double epsilon = 0.000001;

	private Random rand = new Random(1);

	private double lambda = 1;

	private int[] nTopicCounts;

	private ModelMode mode;

	public Synthetic(ModelMode _mode) {
		mode = _mode;
	}

	private double[][] genTopics(int nTopics, int nWords) {
		System.out.println("nTopics = " + nTopics);
		double[][] topics = new double[nTopics][];
		for (int z = 0; z < nTopics; z++) {
			topics[z] = statTool.sampleDirichletSkew(beta, nWords, topicSkewness, mass, rand);
		}
		return topics;
	}

	private double[][] genUserLatentFactors(int nUsers, int nTopics) {
		double[][] userLatentFactor = new double[nUsers][];
		for (int u = 0; u < nUsers; u++) {
			userLatentFactor[u] = statTool.sampleDirichletSkew(alpha, nTopics, userSkewness, mass, rand);
			double min = Double.POSITIVE_INFINITY;
			for (int z = 0; z < nTopics; z++) {
				if (min > userLatentFactor[u][z]) {
					min = userLatentFactor[u][z];
				}
			}

			double norm = 1 / min + 0.1;

			for (int z = 0; z < nTopics; z++) {
				userLatentFactor[u][z] = Math.log(norm * userLatentFactor[u][z]);
				if (userLatentFactor[u][z] < epsilon){
					userLatentFactor[u][z] = epsilon;
				}		
			}
		}
		return userLatentFactor;
	}

	private int[][] genUserActivePlatforms(int nUsers, int nPlatforms, double prop) {
		int[][] activePlatforms = new int[nUsers][nPlatforms];
		int[] platforms = new int[nPlatforms];
		for (int p = 0; p < nPlatforms; p++) {
			platforms[p] = p;
		}
		for (int u = 0; u < nUsers; u++) {
			// randomly shuffle
			for (int i = 0; i < nPlatforms; i++) {
				int p = rand.nextInt(nPlatforms);
				int pp = rand.nextInt(nPlatforms);
				int k = platforms[p];
				platforms[p] = platforms[pp];
				platforms[pp] = k;
			}
			//
			for (int i = 0; i < nPlatforms; i++) {
				activePlatforms[u][i] = 0;
			}
			if (rand.nextDouble() < prop) {
				activePlatforms[u][platforms[0]] = 1;
			} else {
				for (int i = 0; i < nPlatforms; i++) {
					activePlatforms[u][i] = 1;
				}
			}
		}

		return activePlatforms;
	}

	private double[][][] genUserPlatformPreference(int nUsers, int nTopics, int nPlatforms,
			int[][] userActivePlatforms) {
		double[][][] userPlatformPreference = new double[nUsers][nTopics][];
		for (int u = 0; u < nUsers; u++) {
			for (int z = 0; z < nTopics; z++) {
				//userPlatformPreference[u][z] = statTool.sampleDirichletSkew(alpha, nPlatforms, platformSkeness, mass,rand);
				userPlatformPreference[u][z] = new double[nPlatforms];
				userPlatformPreference[u][z][0] = 0.5;
				userPlatformPreference[u][z][1] = 0.5;

				double min = Double.POSITIVE_INFINITY;
				for (int p = 0; p < nPlatforms; p++) {
					if (min > userPlatformPreference[u][z][p]) {
						min = userPlatformPreference[u][z][p];
					}
				}
				double norm = 1 / min + 0.1;

				for (int p = 0; p < nPlatforms; p++) {
					userPlatformPreference[u][z][p] = Math.log(norm * userPlatformPreference[u][z][p]);
				}

				int activePlatform = -1;
				for (int p = 0; p < nPlatforms; p++) {
					if (userActivePlatforms[u][p] == 1) {
						activePlatform = p;
						break;
					}
				}

				for (int p = 0; p < nPlatforms; p++) {
					if (userActivePlatforms[u][p] == 0) {
						userPlatformPreference[u][z][activePlatform] += userPlatformPreference[u][z][p];
						userPlatformPreference[u][z][p] = Double.NEGATIVE_INFINITY;
					}
				}
			}
		}
		return userPlatformPreference;
	}

	private Post genPost(double[] userInterest, double[][] platformPreference, double[][] topics) {
		// #words in the post
		int nTweetWords = rand.nextInt(maxNWords - minNWords) + minNWords;
		Post post = new Post();
		post.words = new int[nTweetWords];
		if (mode == ModelMode.TWITTER_LDA) {
			// topic
			int z = statTool.sampleMult(userInterest, false, rand);
			nTopicCounts[z]++;
			post.topic = z;
			// platform
			post.platform = statTool.sampleMult(platformPreference[z], false, rand);
			// words
			for (int j = 0; j < nTweetWords; j++) {
				post.words[j] = statTool.sampleMult(topics[z], false, rand);
			}
		} else {
			for (int j = 0; j < nTweetWords; j++) {
				// topic
				int z = statTool.sampleMult(userInterest, false, rand);
				// word
				post.words[j] = statTool.sampleMult(topics[z], false, rand);
			}
			// platform is undefined for now
			post.platform = -1;
		}
		return post;
	}

	private double[][] genUserAuthority(int nUsers, int nTopics, double[][] userLatentFactors) {
		double[][] authorities = new double[nUsers][nTopics];
		for (int u = 0; u < nUsers; u++) {
			for (int z = 0; z < nTopics; z++) {
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(sigma, userLatentFactors[u][z] / sigma);
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(omega + userLatentFactors[u][z],
				// userLatentFactors[u][z] / omega);
				/*
				 * GammaDistribution gammaDistribution = new
				 * GammaDistribution(sigma, (userLatentFactors[u][z] / sigma) *
				 * omega);
				 */
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(sigma, Math.sqrt(userLatentFactors[u][z]));
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(sigma + userLatentFactors[u][z],
				// Math.sqrt(userLatentFactors[u][z]));
				
				 //GammaDistribution gammaDistribution = new GammaDistribution(sigma, (userLatentFactors[u][z] / sigma) * omega);
				 //authorities[u][z] = gammaDistribution.sample();
				 
				authorities[u][z] = userLatentFactors[u][z];
				if (authorities[u][z] < epsilon){
					 authorities[u][z] = epsilon;
					}
			}
		}
		return authorities;
	}

	private double[][] genUserHub(int nUsers, int nTopics, double[][] userLatentFactors) {
		double[][] hubs = new double[nUsers][nTopics];
		for (int u = 0; u < nUsers; u++) {
			for (int z = 0; z < nTopics; z++) {
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(delta, userLatentFactors[u][z] / delta);
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(omega + userLatentFactors[u][z] ,
				// userLatentFactors[u][z] / omega);
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(delta,
				// (userLatentFactors[u][z] / delta) * omega);
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(delta, Math.sqrt(userLatentFactors[u][z]));
				// GammaDistribution gammaDistribution = new
				// GammaDistribution(delta + userLatentFactors[u][z],
				// Math.sqrt(userLatentFactors[u][z]));
				// hubs[u][z] = gammaDistribution.sample();
				//GammaDistribution gammaDistribution = new GammaDistribution(delta, (userLatentFactors[u][z] / delta) * omega);
				//hubs[u][z] = gammaDistribution.sample();
				
				hubs[u][z] = userLatentFactors[u][z];
				if (hubs[u][z] < epsilon){
					hubs[u][z] = epsilon;
					}
			}
		}
		return hubs;
	}

	private double getLinkLikelihood(int nTopics, double[] userAuthority, double[][] vPlatformPreference,
			double[] userHub, double[][] uPlatformPreference, int platform) {
		double prod = 0;
		for (int z = 0; z < nTopics; z++) {
			prod += userHub[z] * uPlatformPreference[z][platform] * userAuthority[z] * vPlatformPreference[z][platform];
		}
		prod = MathTool.normalizationFunction(prod * lambda);
		return prod;
	}

	private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> genNetwork(int nUsers, int nTopics, int nPlatforms,
			double[][] userAuthorities, double[][] userHubs, double[][][] userPlatformPreference,
			int[][] userActivePlatforms) {
		double[][][] userRelativePlatformPreference = new double[nUsers][nTopics][];
		for (int u = 0; u < nUsers; u++) {
			for (int z = 0; z < nTopics; z++) {
				userRelativePlatformPreference[u][z] = MathTool.softmax(userPlatformPreference[u][z]);
			}
		}
		HashMap<Integer, HashMap<Integer, HashSet<Integer>>> followings = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
		for (int u = 0; u < nUsers; u++) {
			HashMap<Integer, HashSet<Integer>> uFollowings = new HashMap<Integer, HashSet<Integer>>();
			for (int p = 0; p < nPlatforms; p++) {
				if (userActivePlatforms[u][p] == 0) {
					continue;
				}
				HashSet<Integer> upFollowings = new HashSet<Integer>();
				List<Tuple> tuples = new ArrayList<Tuple>();
				for (int v = 0; v < nUsers; v++) {
					if (u == v)
						continue;
					if (userActivePlatforms[v][p] == 0) {
						continue;
					}
					double prod = getLinkLikelihood(nTopics, userAuthorities[v], userRelativePlatformPreference[v],
							userHubs[u], userRelativePlatformPreference[u], p);
					tuples.add(new Tuple(v, prod));
				}
				Collections.sort(tuples);
				for (int i = 0; i < tuples.size() * 0.2; i++) {
					upFollowings.add(tuples.get(tuples.size() - i - 1).getIntKey());
				}
				uFollowings.put(p, upFollowings);
			}
			followings.put(u, uFollowings);
		}
		return followings;
	}

	private void saveUsers(int nUsers, int[][] userActivePlatforms, String outputPath) {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/users.csv", outputPath)));
			for (int u = 0; u < nUsers; u++) {
				bw.write(String.format("%d,user_%d", u, u));
				for (int p = 0; p < userActivePlatforms[u].length; p++) {
					bw.write(String.format(",%d", userActivePlatforms[u][p]));
				}
				bw.write("\n");
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void saveWords(int nWords, String outputPath) {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/vocabulary.csv", outputPath)));
			for (int w = 0; w < nWords; w++) {
				bw.write(String.format("%d,word_%d\n", w, w));
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void genAndsaveTweet(String outputpath, int nUsers, int nTopics, double[][] userLatentFactors,
			double[][][] userPlatformPreference, double[][] topics) {
		try {

			nTopicCounts = new int[nTopics];

			int nPosts = 0;
			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/posts.csv", outputpath)));
			BufferedWriter bw_topic = new BufferedWriter(
					new FileWriter(String.format("%s/postTopics.csv", outputpath)));
			BufferedWriter bw_empirical = new BufferedWriter(
					new FileWriter(String.format("%s/syn_userEmpiricalTopicDistribution.csv", outputpath)));
			for (int u = 0; u < nUsers; u++) {
				double[] uInterest = MathTool.softmax(userLatentFactors[u]);
				double[][] uPlatformPreference = new double[nTopics][];
				for (int z = 0; z < nTopics; z++) {
					uPlatformPreference[z] = MathTool.softmax(userPlatformPreference[u][z]);
				}
				int n = rand.nextInt(maxNPosts - minNPosts) + minNPosts;

				for (int z = 0; z < nTopics; z++) {
					nTopicCounts[z] = 0;
				}

				for (int i = 0; i < n; i++) {
					Post post = genPost(uInterest, uPlatformPreference, topics);
					bw.write(nPosts + "," + u + "," + post.platform + "," + post.words[0]);
					for (int j = 1; j < post.words.length; j++) {
						bw.write(" " + post.words[j]);
					}
					// batch
					bw.write(",1");
					bw.newLine();
					bw_topic.write(String.format("%d,%d\n", u, post.topic));
					nPosts++;
				}

				bw_empirical.write(String.format("%f", ((double) nTopicCounts[0]) / n));
				for (int z = 1; z < nTopics; z++) {
					bw_empirical.write(String.format(",%f", ((double) nTopicCounts[z]) / n));
				}
				bw_empirical.write("\n");
			}
			bw.close();
			bw_topic.close();
			bw_empirical.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void genAndsaveNetwork(String outputpath, int nUsers, int nPlatforms, int nTopics,
			double[][] userAuthorities, double[][] userHubs, double[][][] userPlatformPreference,
			int[][] userActivePlatforms) {
		try {
			HashMap<Integer, HashMap<Integer, HashSet<Integer>>> followings = genNetwork(nUsers, nTopics, nPlatforms,
					userAuthorities, userHubs, userPlatformPreference, userActivePlatforms);
			// File file = new File(String.format("%s/followings", outputpath));
			// if (!file.exists()) {
			// file.mkdir();
			// }

			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/relationships.csv", outputpath)));
			for (int u = 0; u < nUsers; u++) {
				if (!followings.containsKey(u)) {
					System.out.printf("no-followings u %d\n", u);
					continue;
				}
				for (int p : followings.get(u).keySet()) {
					for (int v : followings.get(u).get(p)) {
						bw.write(u + "," + v + "," + p);
						// batch
						bw.write(",1");
						bw.newLine();
					}
				}
			}
			bw.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void saveGroundTruth(double[][] topics, double[][] userInterest, double[][] userAuthorities,
			double[][] userHubs, double[][][] userPlatformPreference, String outputPath) {
		try {
			BufferedWriter bw;
			String filename = null;

			// topics
			filename = String.format("%s/topicWordDistributions.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			for (int z = 0; z < topics.length; z++) {
				bw.write(String.format("%d", z));
				for (int w = 0; w < topics[z].length; w++) {
					bw.write(String.format(",%f", topics[z][w]));
				}
				bw.write("\n");
			}
			bw.close();

			// user interest
			filename = String.format("%s/userLatentFactors.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			for (int u = 0; u < userInterest.length; u++) {
				bw.write(String.format("%d", u));
				for (int z = 0; z < userInterest[u].length; z++) {
					bw.write(String.format(",%f", userInterest[u][z]));
				}
				bw.write("\n");
			}
			bw.close();

			// user authorities
			filename = String.format("%s/userAuthorities.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			for (int u = 0; u < userAuthorities.length; u++) {
				bw.write(String.format("%d", u));
				for (int z = 0; z < userAuthorities[u].length; z++) {
					bw.write(String.format(",%f", userAuthorities[u][z]));
				}
				bw.write("\n");
			}
			bw.close();

			// user authorities
			filename = String.format("%s/userHubs.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			for (int u = 0; u < userHubs.length; u++) {
				bw.write(String.format("%d", u));
				for (int z = 0; z < userHubs[u].length; z++) {
					bw.write(String.format(",%f", userHubs[u][z]));
				}
				bw.write("\n");
			}
			bw.close();

			// user platform preference
			filename = String.format("%s/userPlatformPreference.csv", outputPath);
			bw = new BufferedWriter(new FileWriter(filename));
			for (int u = 0; u < userPlatformPreference.length; u++) {
				for (int z = 0; z < userPlatformPreference[u].length; z++) {
					bw.write(String.format("%d,%d", u, z));
					for (int p = 0; p < userPlatformPreference[u][z].length; p++) {
						bw.write(String.format(",%f", userPlatformPreference[u][z][p]));
					}
					bw.write("\n");
				}
			}
			bw.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public void genData(int nUsers, int nPlatforms, int nTopics, int nWords, String outputPath) {
		double[][] topics = genTopics(nTopics, nWords);
		double[][] userLatentFactors = genUserLatentFactors(nUsers, nTopics);
		platformSkeness = 1d / nPlatforms;
		//platformSkeness = 1;
		int[][] userActivePlatforms = genUserActivePlatforms(nUsers, nPlatforms, singlePlatformProp);
		double[][][] userPlatformPreference = genUserPlatformPreference(nUsers, nTopics, nPlatforms,
				userActivePlatforms);
		double[][] userAuthorities = genUserAuthority(nUsers, nTopics, userLatentFactors);
		double[][] userHubs = genUserHub(nUsers, nTopics, userLatentFactors);
		saveUsers(nUsers, userActivePlatforms, outputPath);
		saveWords(nWords, outputPath);
		genAndsaveTweet(outputPath, nUsers, nTopics, userLatentFactors, userPlatformPreference, topics);
		genAndsaveNetwork(outputPath, nUsers, nPlatforms, nTopics, userAuthorities, userHubs, userPlatformPreference,
				userActivePlatforms);
		saveGroundTruth(topics, userLatentFactors, userAuthorities, userHubs, userPlatformPreference, outputPath);
	}

	private void testTuple() {
		List<Tuple> tuples = new ArrayList<Tuple>();
		tuples.add(new Tuple(1, 0.1));
		tuples.add(new Tuple(3, 0.3));
		tuples.add(new Tuple(2, 0.2));
		tuples.add(new Tuple(5, 0.5));
		tuples.add(new Tuple(4, 0.4));

		Collections.sort(tuples);
		for (int i = 0; i < tuples.size(); i++) {
			System.out.printf("i = %d key = %d value = %f\n", i, tuples.get(i).getIntKey(),
					tuples.get(i).getDoubleValue());
		}
	}

	public static void main(String[] args) {
		Synthetic generator = new Synthetic(ModelMode.TWITTER_LDA);
		// generator.testTuple();
		//generator.genData(1000, 2, 10, 10000, "E:/code/java/MP-HAT/mp-hat/syn_data");
		 generator.genData(100, 2, 10, 10000, "F:/users/roylee/MP-HAT/mp-hat/syn_data");
		// generator.genData(1000, 2, 10, 1000,
		// "/Users/roylee/Documents/Chardonnay/mp-hat/syn_data/");

	}
}

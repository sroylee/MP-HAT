/***
 * Utilities for synthetic data generation
 */
package data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import org.apache.commons.math3.distribution.GammaDistribution;

import model.Configure.ModelMode;
import tool.MathTool;
import tool.StatTool;

public class Synthetic {

	private class Post {
		public int platform;
		public int[] words;
	}

	private double mass = 0.9;
	private double userSkewness = 0.1;// together with mass, this means, for
										// each user, 90% of her posts are about
										// 10% of topics
	private double topicSkewness = 0.01;// similarly, each topic focuses on 1%
										// of words whose probabilities summing
										// up to 90%
	private double platformSkeness;

	private int minNPosts = 100;
	private int maxNPosts = 200;

	private int minNWords = 10;
	private int maxNWords = 20;

	private StatTool statTool = new StatTool();

	public double alpha = 1;
	public double beta = 1;

	public double sigma = 2;
	public double delta = 2;

	private Random rand = new Random();

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
			}
		}
		return userLatentFactor;
	}

	private double[][][] genUserPlatformPreference(int nUsers, int nTopics, int nPlatforms) {
		double[][][] userPlatformPreference = new double[nUsers][nTopics][];
		for (int u = 0; u < nUsers; u++) {
			for (int z = 0; z < nTopics; z++) {
				userPlatformPreference[u][z] = statTool.sampleDirichletSkew(alpha, nPlatforms, platformSkeness, mass,
						rand);

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
				GammaDistribution gammaDistribution = new GammaDistribution(sigma, userLatentFactors[u][z] / sigma);
				authorities[u][z] = gammaDistribution.sample();
			}
		}
		return authorities;
	}

	private double[][] genUserHub(int nUsers, int nTopics, double[][] userLatentFactors) {
		double[][] hubs = new double[nUsers][nTopics];
		for (int u = 0; u < nUsers; u++) {
			for (int z = 0; z < nTopics; z++) {
				GammaDistribution gammaDistribution = new GammaDistribution(delta, userLatentFactors[u][z] / delta);
				hubs[u][z] = gammaDistribution.sample();
			}
		}
		return hubs;
	}

	private int genLink(int nTopics, double[] userAuthority, double[][] vPlatformPreference, double[] userHub,
			double[][] uPlatformPreference, int platform) {
		double prod = 0;
		for (int z = 0; z < nTopics; z++) {
			prod = userHub[z] * uPlatformPreference[z][platform] * userAuthority[z] * vPlatformPreference[z][platform];
		}

		prod = MathTool.normalizationFunction(prod);
		// System.out.println("p = " + p);
		if (rand.nextDouble() < prod) {
			return 1;
		}
		return 0;
	}

	private HashMap<Integer, HashMap<Integer, HashSet<Integer>>> genNetwork(int nUsers, int nTopics, int nPlatforms,
			double[][] userAuthorities, double[][] userHubs, double[][][] userPlatformPreference) {
		HashMap<Integer, HashMap<Integer, HashSet<Integer>>> followings = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();
		for (int u = 0; u < nUsers; u++) {
			HashMap<Integer, HashSet<Integer>> uFollowings = new HashMap<Integer, HashSet<Integer>>();
			for (int p = 0; p < nPlatforms; p++) {
				HashSet<Integer> upFollowings = new HashSet<Integer>();
				for (int v = 0; v < nUsers; v++) {
					if (u == v)
						continue;
					int link = genLink(nTopics, userAuthorities[v], userPlatformPreference[v], userHubs[u],
							userPlatformPreference[u], p);
					if (link == 1) {
						upFollowings.add(v);
					}
				}
				uFollowings.put(p, upFollowings);
			}
			followings.put(u, uFollowings);
		}
		return followings;
	}

	private void saveUsers(int nUsers, String outputPath) {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/syn_users.csv", outputPath)));
			for (int u = 0; u < nUsers; u++) {
				bw.write(String.format("%d,user_%d\n", u, u));
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void saveWords(int nWords, String outputPath) {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/syn_vocabulary.csv", outputPath)));
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
			BufferedWriter bw = new BufferedWriter(new FileWriter(String.format("%s/syn_posts.csv", outputpath)));
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
					nPosts++;
				}

				bw_empirical.write(String.format("%f", ((double) nTopicCounts[0]) / n));
				for (int z = 1; z < nTopics; z++) {
					bw_empirical.write(String.format(",%f", ((double) nTopicCounts[z]) / n));
				}
				bw_empirical.write("\n");
			}
			bw.close();
			bw_empirical.close();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	private void genAndsaveNetwork(String outputpath, int nUsers, int nPlatforms, int nTopics,
			double[][] userAuthorities, double[][] userHubs, double[][][] userPlatformPreference) {
		try {
			HashMap<Integer, HashMap<Integer, HashSet<Integer>>> followings = genNetwork(nUsers, nTopics, nPlatforms,
					userAuthorities, userHubs, userPlatformPreference);
			// File file = new File(String.format("%s/followings", outputpath));
			// if (!file.exists()) {
			// file.mkdir();
			// }

			BufferedWriter bw = new BufferedWriter(
					new FileWriter(String.format("%s/syn_relationships.csv", outputpath)));
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
		platformSkeness = 1d / nTopics;
		double[][][] userPlatformPreference = genUserPlatformPreference(nUsers, nTopics, nPlatforms);
		double[][] userAuthorities = genUserAuthority(nUsers, nTopics, userLatentFactors);
		double[][] userHubs = genUserHub(nUsers, nTopics, userLatentFactors);
		saveUsers(nUsers, outputPath);
		saveWords(nWords, outputPath);
		genAndsaveTweet(outputPath, nUsers, nTopics, userLatentFactors, userPlatformPreference, topics);
		genAndsaveNetwork(outputPath, nUsers, nPlatforms, nTopics, userAuthorities, userHubs, userPlatformPreference);
		saveGroundTruth(topics, userLatentFactors, userAuthorities, userHubs, userPlatformPreference, outputPath);
	}

	public static void main(String[] args) {
		Synthetic generator = new Synthetic(ModelMode.TWITTER_LDA);
		//generator.genData(1000, 2, 10, 1000, "E:/code/java/MP-HAT/mp-hat/output");
		generator.genData(1000, 2, 10, 1000, "E:/users/roylee.2013/Chardonnay/synthetic");
	}
}

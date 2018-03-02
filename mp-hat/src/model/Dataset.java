package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.PriorityBlockingQueue;

import tool.KeyValuePair;

public class Dataset {
	public String path;
	public int nUsers;
	public int[] nPlatformUsers;
	public int nPlatforms;
	public User[] users;
	public int nLinks = 0;
	public int nNonLinks = 0;

	// for selecting non-links
	private KeyValuePair[] userRankByNFollowers;
	private KeyValuePair[] userRankByNFollowings;

	// public int nWords; // number of words in vocabulary
	public String[] vocabulary;

	public HashMap<String, Integer> userId2Index;
	public HashMap<Integer, String> userIndex2Id;

	/***
	 * read dataset from folder "path"
	 * 
	 * @param path
	 */
	public Dataset(String _path, int batch, boolean onlyLearnGibbs) {
		this.path = _path;
		System.out.println("loading user list");
		loadUsers(String.format("%s/users.csv", path));
		System.out.println("loading posts");
		loadPosts(String.format("%s/posts.csv", path));
		System.out.println("loading vocabulary");
		loadVocabulary(String.format("%s/vocabulary.csv", path));
		
		if (onlyLearnGibbs==false){
			System.out.println("loading links");
			loadRelationship(String.format("%s/relationships.csv", path));
			// System.out.println("loading non-links");
			// loadNonRelationship(String.format("%s/nonrelationships.csv", path));
			for (int p=0; p<Configure.NUM_OF_PLATFORM;p++){
				selectNonRelationship(batch,p);
			}
			System.out.println("#Links:" + nLinks);
			System.out.println("#NonLinks:" + nNonLinks);
		}
	}

	private void loadUsers(String filename) {
		Scanner sc = null;
		BufferedReader br = null;
		String line = null;
		userId2Index = new HashMap<String, Integer>();
		userIndex2Id = new HashMap<Integer, String>();

		try {
			File userFile = new File(filename);

			// Get total number of users and initiate the Users array
			br = new BufferedReader(new FileReader(userFile.getAbsolutePath()));
			while (br.readLine() != null) {
				nUsers++;
			}
			br.close();

			// Declare the number of users in users array
			users = new User[nUsers];

			// Read and load user into Users array
			br = new BufferedReader(new FileReader(userFile.getAbsolutePath()));
			int u = 0;
			while ((line = br.readLine()) != null) {
				sc = new Scanner(line.toString());
				sc.useDelimiter(",");
				while (sc.hasNext()) {
					String userId = sc.next();
					String username = sc.next();
					users[u] = new User();
					users[u].userId = userId.trim();
					users[u].username = username;
					users[u].userIndex = u;
					userId2Index.put(userId, u);
					userIndex2Id.put(u, userId);
					//Check the user has account with which platform (each column represent a platform. 1 represent present, 0 otherwise
					for (int p=0; p<Configure.NUM_OF_PLATFORM;p++){
						int flag = sc.nextInt();
						nPlatformUsers[p] += flag;
						users[u].platforms[p] = flag;
					}
					u++;
				}
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void loadPosts(String filename) {

		BufferedReader br = null;
		String line = null;

		try {
			File file = new File(filename);

			// Get total number of users' posts
			for (int u = 0; u < nUsers; u++) {
				users[u].nPosts = 0;
			}

			br = new BufferedReader(new FileReader(file.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String userId = line.split(",")[1];
				int u = userId2Index.get(userId);
				users[u].nPosts++;
			}

			br.close();

			// initalize the users'post arrays
			for (int u = 0; u < nUsers; u++) {
				users[u].posts = new Post[users[u].nPosts];
				users[u].postBatches = new int[users[u].nPosts];
				users[u].nPosts = 0;
			}

			// Read and load user into users' follower and following array
			br = new BufferedReader(new FileReader(file.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String postId = tokens[0];
				String userId = tokens[1];
				int platform = Integer.parseInt(tokens[2]);
				int batch = Integer.parseInt(tokens[3]);
				int u = userId2Index.get(userId);
				users[u].postBatches[users[u].nPosts] = batch;
				users[u].posts[users[u].nPosts] = new Post();
				users[u].posts[users[u].nPosts].postId = postId;
				tokens = tokens[2].trim().split(" ");
				users[u].posts[users[u].nPosts].nWords = tokens.length;
				users[u].posts[users[u].nPosts].words = new int[tokens.length];
				for (int i = 0; i < tokens.length; i++) {
					// System.out.println(postId+","+tokens[i]);
					users[u].posts[users[u].nPosts].words[i] = Integer.parseInt(tokens[i]);

				}
				users[u].posts[users[u].nPosts].platform = platform;
				users[u].nPosts++;
				users[u].nPlatformPosts[platform]++;
			}
			br.close();
		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void loadVocabulary(String filename) {
		BufferedReader br = null;
		String line = null;

		try {
			br = new BufferedReader(new FileReader(filename));
			int nVocabs = 0;
			while (br.readLine() != null) {
				nVocabs++;
			}
			br.close();
			vocabulary = new String[nVocabs];

			br = new BufferedReader(new FileReader(filename));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				int index = Integer.parseInt(tokens[0]);
				String vocab = tokens[1];
				vocabulary[index] = vocab;
			}
			br.close();
			// System.out.println("Number of Vocabulary loaded:" + nVocabs);
		} catch (Exception e) {
			System.out.println("Error in reading vocabulary file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void loadRelationship(String filename) {
		BufferedReader br = null;
		String line = null;

		for (int u = 0; u < nUsers; u++) {
			users[u].nFollowers = 0;
			users[u].nFollowings = 0;
		}

		try {
			File file = new File(filename);

			// Get total number of users' followers and following
			br = new BufferedReader(new FileReader(file.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String src_user = tokens[0];
				String des_user = tokens[1];
				int src_user_index = userId2Index.get(src_user);
				int des_user_index = userId2Index.get(des_user);
				// update follower count
				users[des_user_index].nFollowers++;
				// update following count
				users[src_user_index].nFollowings++;

			}
			br.close();

			// initalize the users' follower and following arrays
			for (int u = 0; u < nUsers; u++) {
				if (users[u].nFollowers > 0) {
					users[u].followers = new Follower[users[u].nFollowers];
					users[u].followerBatches = new int[users[u].nFollowers];
					users[u].nFollowers = 0;
				}
				if (users[u].nFollowings > 0) {
					users[u].followings = new Following[users[u].nFollowings];
					users[u].followingBatches = new int[users[u].nFollowings];
					users[u].nFollowings = 0;
				}
			}
			
			// Read and load user into users' follower and following array
			br = new BufferedReader(new FileReader(file.getAbsolutePath()));
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				String src_user = tokens[0];
				String des_user = tokens[1];
				int platform = Integer.parseInt(tokens[2]);
				int batch = Integer.parseInt(tokens[3]);

				int src_user_index = userId2Index.get(src_user);
				int des_user_index = userId2Index.get(des_user);
				users[des_user_index].followers[users[des_user_index].nFollowers].followerIndex = src_user_index;
				users[des_user_index].followers[users[des_user_index].nFollowers].platform = platform;
				users[des_user_index].followerBatches[users[des_user_index].nFollowers] = batch;
				users[des_user_index].nFollowers++;
				users[des_user_index].nPlatformFollowers[platform]++;
				

				users[src_user_index].followings[users[src_user_index].nFollowings].followingIndex = des_user_index;
				users[src_user_index].followings[users[src_user_index].nFollowings].platform = platform;
				users[src_user_index].followingBatches[users[src_user_index].nFollowings] = batch;
				users[src_user_index].nFollowings++;
				users[src_user_index].nPlatformFollowings[platform]++;
				if (batch == 1) {
					nLinks++;
				}
			}
			br.close();

		} catch (Exception e) {
			System.out.println("Error in reading user file!");
			e.printStackTrace();
			System.exit(0);
		}
	}

	private void rankUserbyPopuarlity(int platform) {
		userRankByNFollowers = new KeyValuePair[nUsers];
		userRankByNFollowings = new KeyValuePair[nUsers];
		for (int u = 0; u < nUsers; u++) {
			userRankByNFollowers[u] = new KeyValuePair(u, users[u].nFollowers);
			userRankByNFollowings[u] = new KeyValuePair(u, users[u].nFollowings);
		}
		Arrays.sort(userRankByNFollowers);
		Arrays.sort(userRankByNFollowings);
	}

	public void selectNonRelationship(int batch, int platform) {
		rankUserbyPopuarlity(platform);

		int[] userNonFollowerCounts = new int[nPlatformUsers[platform]];
		int[] maxNonFollowers = new int[nPlatformUsers[platform]];

		HashMap<Integer, HashSet<Integer>> userNonFollowers = new HashMap<Integer, HashSet<Integer>>();

		for (int u = 0; u < nUsers; u++) {
			maxNonFollowers[u] = (int) ((nUsers - users[u].nPlatformFollowers[platform] - 1) * Configure.PROPTION_OF_NONLINKS);
			userNonFollowerCounts[u] = 0;
		}

		for (int r = 0; r < nUsers; r++) {
			int u = userRankByNFollowings[r].getIntKey();
			// this will make most of the non-links are from less-followees
			// users to many-followers (e.g., popular) users

			// get followee set
			HashSet<Integer> followings = new HashSet<Integer>();
			for (int i = 0; i < users[u].nFollowings; i++) {
				if (users[u].followingBatches[i] != batch) {
					continue;
				}
				if (users[u].followings[i].platform != platform){
					continue;
				}
				followings.add(users[u].followings[i].followingIndex);
			}

			// #selected non-followings:
			int nNonFollowings = (int) ((nPlatformUsers[platform] - users[u].nPlatformFollowings[platform] - 1) * Configure.PROPTION_OF_NONLINKS);

			// select non-followings
			HashSet<Integer> nonfollwings = new HashSet<Integer>();
			// (1): select from popular users
			int nPopularUsers = (int) (nPlatformUsers[platform] * Configure.PROPTION_OF_POPULAR_USERS);
			for (int i = 0; i < nPopularUsers; i++) {
				int v = userRankByNFollowers[nUsers - i - 1].getIntKey();
				if (v == u) {
					continue;
				}
				if (followings.contains(v)) {
					continue;
				}
				if (userNonFollowerCounts[v] >= maxNonFollowers[v]) {
					continue;
				}
				if (users[v].platforms[platform]==0){
					continue;
				}

				nonfollwings.add(v);
				nNonFollowings--;

				userNonFollowerCounts[v]++;

				if (userNonFollowers.containsKey(v)) {
					userNonFollowers.get(v).add(u);
				} else {
					HashSet<Integer> nonFollowers = new HashSet<Integer>();
					nonFollowers.add(u);
					userNonFollowers.put(v, nonFollowers);
				}

				if (nNonFollowings == 0) {
					break;
				}
			}

			// (2): if not enough, select the remaining from top non-followees
			// of followees
			if (nNonFollowings > 0) {
				// get nonfollwees among followees of followees
				HashMap<Integer, Integer> followeesOfFollowees = new HashMap<Integer, Integer>();
				for (int i = 0; i < users[u].nFollowings; i++) {
					if (users[u].followingBatches[i] != batch) {
						continue;
					}
					if (users[u].followings[i].platform != platform) {
						continue;
					}
					int v = users[u].followings[i].followingIndex;
					for (int j = 0; j < users[v].nFollowings; j++) {
						if (users[v].followingBatches[j] != batch) {
							continue;
						}
						if (users[v].followings[j].platform != platform) {
							continue;
						}
						int w = users[v].followings[j].followingIndex;
						if (w == u) {
							continue;
						}
						if (followings.contains(w) || users[u].userId.equals(users[w].userId)) {
							continue;
						}
						if (users[w].platforms[platform] != 1){
							continue;
						}
						if (followeesOfFollowees.containsKey(w)) {
							followeesOfFollowees.put(w, 1 + followeesOfFollowees.get(w));
						} else {
							followeesOfFollowees.put(w, 1);
						}
					}
				}
				// rank by #intermediate followees
				PriorityBlockingQueue<KeyValuePair> queue = new PriorityBlockingQueue<KeyValuePair>();
				for (Map.Entry<Integer, Integer> pair : followeesOfFollowees.entrySet()) {

					if (nonfollwings.contains(pair.getKey())) {
						// already among the popular users
						continue;
					}

					if (queue.size() < nNonFollowings) {
						queue.add(new KeyValuePair(pair.getKey(), pair.getValue()));
					} else {
						KeyValuePair head = queue.peek();
						if (head.getIntValue() < pair.getValue()) {
							queue.poll();
							queue.add(new KeyValuePair(pair.getKey(), pair.getValue()));
						}
					}
				}
				// add into selected list
				while (!queue.isEmpty()) {
					int v = queue.poll().getIntKey();
					if (userNonFollowerCounts[v] >= maxNonFollowers[v]) {
						continue;
					}
					userNonFollowerCounts[v]++;
					if (userNonFollowers.containsKey(v)) {
						userNonFollowers.get(v).add(u);
					} else {
						HashSet<Integer> nonFollowers = new HashSet<Integer>();
						nonFollowers.add(u);
						userNonFollowers.put(v, nonFollowers);
					}
					nonfollwings.add(v);
					nNonFollowings--;
					if (nNonFollowings == 0) {
						break;
					}
				}
			}
			// (3): if still not enough, continue to select from less popular
			// users
			if (nNonFollowings > 0) {
				for (int i = nPopularUsers; i < nUsers; i++) {
					int v = userRankByNFollowers[nPlatformUsers[platform] - i - 1].getIntKey();
					if (v == u) {
						continue;
					}
					if (followings.contains(v) || users[u].userId.equals(users[v].userId)) {
						continue;
					}
					if (userNonFollowerCounts[v] >= maxNonFollowers[v]) {
						continue;
					}
					if (users[v].platforms[platform] !=1){
						continue;
					}
					userNonFollowerCounts[v]++;
					if (userNonFollowers.containsKey(v)) {
						userNonFollowers.get(v).add(u);
					} else {
						HashSet<Integer> nonFollowers = new HashSet<Integer>();
						nonFollowers.add(u);
						userNonFollowers.put(v, nonFollowers);
					}
					nonfollwings.add(v);
					nNonFollowings--;
					if (nNonFollowings == 0) {
						break;
					}
				}
			}

			// add into user's non-followee list
			users[u].nNonFollowings = 0;
			users[u].nonFollowings = new Following[nonfollwings.size()];
			users[u].nonFollowingBatches = new int[nonfollwings.size()];
			for (int v : nonfollwings) {
				users[u].nonFollowings[users[u].nNonFollowings].followingIndex = v;
				users[u].nonFollowings[users[u].nNonFollowings].platform = platform;
				users[u].nonFollowingBatches[users[u].nNonFollowings] = 1;
				users[u].nPlatformNonFollowings[platform]++;
				users[u].nNonFollowings++;
				nNonLinks++;
			}
		}

		// Reverse infer the non-followers from the non-following
		for (int v = 0; v < users.length; v++) {
			HashSet<Integer> nonFollowers = userNonFollowers.get(v);
			if (nonFollowers == null) {
				continue;
			}
			users[v].nonFollowers = new Follower[nonFollowers.size()];
			users[v].nNonFollowers = 0;
			for (int u : nonFollowers) {
				users[v].nonFollowers[users[v].nNonFollowers].followerIndex = u;
				users[v].nonFollowers[users[v].nNonFollowers].platform = platform;
				users[v].nPlatformNonFollowers[platform]++;
				users[v].nNonFollowers++;
			}
			//System.out.println(users[v].userId + " " + users[v].nonFollowers.length);
		}

	}

	public static void main(String[] args) {
		KeyValuePair[] x = new KeyValuePair[10];
		for (int i = 0; i < 10; i++) {
			x[i] = new KeyValuePair(i, i * 2 % 5);
		}
		Arrays.sort(x);
		for (int i = 0; i < 10; i++) {
			System.out.printf("x[%d] = (%d, %d)\n", i, x[i].getIntKey(), x[i].getIntValue());
		}
	}

	public void output_NonLinks() {
		try {
			File f = new File(path + "/" + "l_generatedNonLinks.csv");
			FileWriter fo = new FileWriter(f);
			for (int u = 0; u < users.length; u++) {
				String uid = users[u].userId;
				for (int v = 0; v < users[u].nonFollowings.length; v++) {
					String vid = users[users[u].nonFollowings[v].followingIndex].userId;
					int platform = users[u].nonFollowings[v].platform;
					fo.write(uid + "," + vid + "," + platform + "\n");
				}
			}
			fo.close();
		} catch (Exception e) {
			System.out.println("Error in writing to topical interest file!");
			e.printStackTrace();
			System.exit(0);
		}
	}
}

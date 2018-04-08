package model;

public class User {
	public int userIndex;
	public String userId;
	public String username;
	public int[] platforms;

	public int nPosts;
	public int[] nPlatformPosts;
	public Post[] posts; // posts of of the users from multiple platforms
	public int[] postBatches;// batch index of posts, to be used for K-fold
								// cross validation

	public int nFollowings;
	public int[] nPlatformFollowings;
	public Following[] followings;// followings of the user in multiple
									// platforms
	public int[] followingBatches;// batch index of followings, to be used for
									// K-fold cross validation.
									// note that the batch is proportion to the
									// followings in each platform

	public int nNonFollowings;
	public int[] nPlatformNonFollowings;
	public Following[] nonFollowings;// non_followings of the user in multiple
										// platforms
	public int[] nonFollowingBatches;// batch index of non-followings, to be
										// used for K-fold cross validation.
										// note that the batch is proportion to
										// the followings in each platform

	public int nFollowers;
	public int[] nPlatformFollowers;
	public Follower[] followers;// followers of the user in multiple platforms
	public int[] followerBatches;// batch index of followers, to be used for
									// K-fold cross validation.

	public int nNonFollowers;
	public int[] nPlatformNonFollowers;
	public Follower[] nonFollowers; // non_followers of the user in multiple
									// platforms

	public double[] topicalInterests;// theta, K-topics dimension
	public double[] authorities;// A, K-topics dimension
	public double[] hubs;// H, K-topics dimension

	public double[][] topicalPlatformPreference; // eta, K-topics by P-platforms
													// dimension
	public double[][] topicalRelativePlatformPreference; // softmax(eta)

	public double[] optTopicalInterests;// optimized theta, K-topics dimension
	public double[] optAuthorities;// optimized A, K-topics dimension
	public double[] optHubs;// optimized H, K-topics dimension

	public double[][] optTopicalPlatformPreference; // eta, K-topics by
													// P-platforms dimension
	public double[][] optPlatformAuthorities;// optimized A, P-platforms by
												// K-topics dimension
	public double[][] optPlatformHubs;// optimized H, P-platforms by K-topics
										// dimension

	// groundtruth
	public double[] groundtruth_TopicalInterests;
	public double[] groundtruth_Authorities;
	public double[] groundtruth_Hubs;
	public double[][] groundtruth_TopicalPlatformPreference;
}

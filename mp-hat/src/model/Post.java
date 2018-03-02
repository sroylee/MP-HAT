package model;

public class Post {
	public String postId;
	public int nWords;
	public int[] words;// index of words in vocabulary
	public int topic; // assume each post only have one topic
	public int[] wordTopics; // assume each post have multiple topics
	public int platform; // platform of the post
}

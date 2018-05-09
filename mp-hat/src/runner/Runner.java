package runner;

import data.Synthetic;
import evaluation.Prediction;
import model.MultiThreadMPHAT;
import model.Configure.ModelMode;
import model.Configure.PredictionMode;

public class Runner {

	static void syntheticDataGeneration(int nUsers, int nPlatform, int nTopics, int nWords, ModelMode mode,
			String outputPath) {
		data.Synthetic synthetic = new Synthetic(mode);
		synthetic.genData(nUsers, nPlatform, nTopics, nWords, outputPath);
	}

	static void runMPHAT(String datasetPath, int nTopics, int batch, String outputPath) {
		model.MultiThreadMPHAT model = new MultiThreadMPHAT(datasetPath, nTopics, batch, outputPath);
		model.train();
	}

	static void predict(String datasetPath, String resultPath, int nTopics, int nPlatforms, int testBatch,
			PredictionMode predMode, String outputPath) {
		evaluation.Prediction prediction = new Prediction(datasetPath, resultPath, nTopics, nPlatforms, testBatch,
				predMode, outputPath);
		prediction.evaluate();
	}

	static void test() {
		// String datasetPath = "E:/code/java/ctlr/data/acm";
		// int nTopics = 10;
		// int batch = 1;
		// ModelMode mode = ModelMode.TWITTER_LDA;
		// larc.ctlr.model.MultithreadCTLR model = new
		// MultithreadCTLR(datasetPath, nTopics, batch, mode);
		// model.init();
		// model.train();
	}

	public static void main(String[] args) {
		try {
			if (args[0].equals("gen")) {
				int nUsers = Integer.parseInt(args[1]);
				int nPlatforms = Integer.parseInt(args[2]);
				int nTopics = Integer.parseInt(args[3]);
				int nWords = Integer.parseInt(args[4]);
				int mode = Integer.parseInt(args[5]);
				String outputPath = args[6];
				if (mode == 0) {
					syntheticDataGeneration(nUsers, nPlatforms, nTopics, nWords, ModelMode.TWITTER_LDA, outputPath);
				} else {
					syntheticDataGeneration(nUsers, nPlatforms, nTopics, nWords, ModelMode.ORIGINAL_LDA, outputPath);
				}
			} else if (args[0].equals("mphat")) {
				String datasetPath = args[1];
				int nTopics = Integer.parseInt(args[2]);
				int batch = Integer.parseInt(args[3]);
				String outputPath = args[4];
				runMPHAT(datasetPath, nTopics, batch, outputPath);
			} else if (args[0].equals("predict")) {
				String datasetPath = args[1];
				String resultPath = args[2];
				int topics = Integer.parseInt(args[3]);
				int platforms = Integer.parseInt(args[4]);
				int testBatch = Integer.parseInt(args[5]);
				int predMode = Integer.parseInt(args[6]);
				String outputPath = args[7];
				if (predMode == 0) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.HAT, outputPath);
				} else if (predMode == 1) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.COMMON_INTEREST,
							outputPath);
				} else if (predMode == 2) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.COMMON_NEIGHBOR,
							outputPath);
				} else if (predMode == 3) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.HITS, outputPath);
				} else if (predMode == 4) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.CTR, outputPath);
				} else if (predMode == 5) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.WTFW, outputPath);
				} else if (predMode == 6) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.MPHAT, outputPath);
				} else if (predMode == 7) {
					predict(datasetPath, resultPath, topics, platforms, testBatch, PredictionMode.MPHAT_TI, outputPath);
				}
			} else if (args[0].equals("hits")) {
				// String datasetPath = args[1];
				// int batch = Integer.parseInt(args[2]);
				// hits(datasetPath, batch);
			} else {
				System.out.printf("%s is not an option!!!");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
const tf = require("@tensorflow/tfjs");
const _ = require("lodash");


class LinearRegression {

	constructor(features, labels, options) {
		this.labels = tf.tensor(labels);
		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000,
		}, options);

		this.features = this.processFeatures(features);
		this.features = tf.one([this.features.shape[0], 1])
			.concat(this.features, 1);

		// gradient descent vars
		this.weights = tf.zeros([this.features.shape[1], 1]);
		this.mseHistory = [];
		this.mean = null;
		this.variance = null;
	}

	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent();
			this.recordMSE();
			this.updateLearningRate();
		}
	}

	gradientDescent() {
		const guesses = this.features.matMul(this.weights);
		const differences = guesses.sub(this.labels);
		const slopes = this.features
			.transpose()
			.matMul(differences)
			.div(this.features.shape[0]);
		this.weights = this.weights
			.sub(slopes.mul(this.options.learningRate));
	}

	standardize(features) {
		const { mean, variance } = tf.moments(features, 0);
		this.mean = mean;
		this.variance = variance;
	}

	processFeatures(features) {
		features = tf.tensor(features)
		if (this.mean === null && this.variance === null) {
			this.standardize(features)
		}
		features = features.sub(this.mean).div(this.variance.pow(0.5));

		features = tf.ones(features.shape[0], 1).concat(features, 1);
		return features;
	}

	updateLearningRate() {
		if (this.mseHistory.length < 2) return;
		const [ first, second ] = this.mseHistory;
		if (first > second) {
			this.options.learningRate /= 2;
			return
		}
		this.options.learningRate *= 0.5;
	}

	recordMSE() {
		const mse = this.features.matMul(this.weights)
			.sub(this.labels).pow(2)
			.sum().div(this.features.shape[0]).get();
		this.mseHistory.unshift(mse);
	}

	test(testFeatures, testLabels) {
		testLabels = tf.tensor(testLabels);
		testFeatures = this.processFeatures(testFeatures);
		const predictions = testFeatures.matMul(this.weights);
		const ss_res = testLabels.sub(predictions).pow(2).sum().get();
		const ss_tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();
		return 1 - (ss_res / ss_tot);
	}

	predict() {

	}
}

module.exports = LinearRegression

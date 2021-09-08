const tf = require("@tensorflow/tfjs");
const _ = require("lodash");


// LinearRegression with gradient descent using Batches
class LinearRegressionWithBGD {

	constructor(features, labels, options) {
		this.labels = tf.tensor(labels);
		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000,
			batchSize: 10,
		}, options);

		this.features = this.processFeatures(features);
		this.features = tf.ones([this.features.shape[0], 1])
			.concat(this.features, 1);

		// gradient descent vars
		this.weights = tf.zeros([this.features.shape[1], 1]);
		this.mseHistory = [];
		this.mean = null;
		this.variance = null;
	}

	train() {
		const batchSize = this.options.batchSize;
		let batchQuantity = math.floor(this.features.shape[0] / batchSize);
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const start = j * batchSize;
				const count = this.options.batchSize;
				this.gradientDescent(
					this.features.slice([start, 0], [count, -1]),
					this.labels.slice([start, 0], [count, -1]),
				);
			}
			this.recordMSE();
			this.updateLearningRate();
		}
	}

	gradientDescent(features, labels) {
		const guesses = features.matMul(this.weights);
		const differences = guesses.sub(labels);
		const slopes = features
			.transpose()
			.matMul(differences)
			.div(features.shape[0]);
		this.weights = this.weights
			.sub(slopes.mul(tf.tensor(this.options.learningRate)));
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
			.sub(this.labels).pow(tf.tensor(2))
			.sum().div(this.features.shape[0]).get();
		this.mseHistory.unshift(mse);
	}

	test(testFeatures, testLabels) {
		testLabels = tf.tensor(testLabels);
		testFeatures = this.processFeatures(testFeatures);
		const predictions = testFeatures.matMul(this.weights);
		// where ss_res => summed_squared_residual
		const ss_res = testLabels.sub(predictions).pow(tf.tensor(2)).sum().get();
		// where ss_total => summed_squared_total
		const ss_tot = testLabels.sub(testLabels.mean()).pow(tf.tensor(2)).sum().get();
		return 1 - (ss_res / ss_tot);
	}

	predict(observations) {
		return this.processFeatures(observations)
			.matMul(this.weights);
	}
}

module.exports = LinearRegressionWithBGD

require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

// LinearRegression with gradient descent using Batches
class MultiNominalLogisticRegression {
	constructor(features, labels, options) {
		this.labels = tf.tensor(labels);
		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000,
			batchSize: 10,
			decisionBoundary: 0.5,
		}, options);

		this.features = this.processFeatures(features);
		this.features = tf.ones([this.features.shape[0], 1])
			.concat(this.features, 1);

		// gradient descent vars
		this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
		this.crossEntropyHistory = [];
		this.mean = null;
		this.variance = null;
	}

	train() {
		const batchSize = this.options.batchSize;
		let batchQuantity = math.floor(this.features.shape[0] / this.options.batchSize);
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const start = j * batchSize;
				const count = this.options.batchSize;
				this.weights = tf.tidy(() => {
					return this.sigmoidEqGradientDescend(
						this.features.slice([start, 0], [count, -1]),
						this.labels.slice([start, 0], [count, -1]),
					);

				});
			}
			this.recordCrossEntropy();
			this.updateLearningRate();
		}
	}

	sigmoidEqGradientDescend(features, labels) {
		// use softmax instead of sigmoid to ensure we do a Conditional Probability Distribution;
		// as sigmoid performs a marginal probability distribution that considers each value
		// of m individual not as a whole.
		const guesses = features.matMul(this.weights).softmax()
		const differences = guesses.sub(labels);
		const slopes = features
			.transpose()
			.matMul(differences)
			.div(features.shape[0]);
		 return this.weights
			.sub(slopes.mul(tf.tensor(this.options.learningRate)));
	}

	standardize(features) {
		const { mean, variance } = tf.tidy(() => {
			const {mean, variance} = tf.moments(features, 0);
			const varianceMask = variance.cast('bool').logicalNot().cast('float32');
			const newVariance = variance.add(varianceMask)
			// major change: variance could end up doing a division by zero which turns into a NaN,
			// we will use the following trick to create a mask where we get rid of the zeros by swapping
			// them with a 1, since division by 1 wont affect the actual result of the calculation.
			return { mean, variance: newVariance };
		});
		this.mean = mean;
		this.variance = variance;
	}

	processFeatures(features) {
		return tf.tidy(() => {
			features = tf.tensor(features)
			if (this.mean === null && this.variance === null) {
				this.standardize(features)
			}
			features = features.sub(this.mean).div(this.variance.pow(0.5));

			return tf.ones(features.shape[0], 1).concat(features, 1);
		});
	}

	updateLearningRate() {
		if (this.crossEntropyHistory.length < 2) return;
		const [ first, second ] = this.crossEntropyHistory;
		if (first > second) {
			this.options.learningRate /= 2;
			return
		}
		this.options.learningRate *= 0.5;
	}

	recordCrossEntropy() {
		// use vectorized cross entropy equation (replaced of mse in linear regression with gradient descent when doing logistic regression):
		// => (-(1/m) . (Actual(T).log(Guesses) + (1-Actual)(T). log(1 - Guesses)))
		// we can do a 1-m to an operation where we do => (m * -1) + 1
		this.crossEntropyHistory.unshift(tf.tidy(() => {
			const guesses = this.features.matMul(this.weights).sigmoid();
			const termOne  = this.labels.transpose().matMul(guesses.add(tf.tensor(1e-7)).log());
			const termTwo = this.labels
				.mul(tf.tensor(-1))
				.add(tf.tensor(1))
				.transpose()
				.matMul(
					guesses
						.mul(tf.tensor(-1))
						.add(tf.tensor(1))
						.add(tf.tensor(1e-7)) // add constant to avoid NaN when we do a log of 0.
						.log(),
				);

			return termOne.add(termTwo)
				.div(this.features.shape[0])
				.mul(tf.tensor(-1))
				.get(0, 0);
		}));
	}

	test(testFeatures, testLabels) {
		return tf.tidy(() => {
			testLabels = tf.tensor(testLabels).argMax(1);

			const predictions = this.predict(testFeatures);
			const differences = predictions.notEqual(testLabels);
			const incorrect = differences.sum().get();

			// divide predictions by incorrect scalar and divide by total number of predictions.
			return (predictions.shape[0] - incorrect) / predictions.shape[0];
		}).get();
	}

	predict(observations) {
		// use softmax instead of sigmoid to ensure we do a Conditional Probability Distribution;
		// as sigmoid performs a marginal probability distribution that considers each value
		// of m individual not as a whole.
		return this.processFeatures(observations)
			.matMul(this.weights)
			.softmax()
			.argMax(1)
	}
}

module.exports = MultiNominalLogisticRegression

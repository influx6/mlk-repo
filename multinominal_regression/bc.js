require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

const readBooks = 1 // => reading books
const watchMovies = 0 // => watching movies

/* step 1. encode your choices into numeric form (binary form if two: 0 and 1)
const dataset = [
	[5, watchMovies],
	[15, watchMovies],
	[25, readBooks],
	[35, readBooks],
	[45, readBooks],
]

step 2:
 use a different equation from linear regression where we can ensure the values
 are between our encoded ranges:

 eq => 1 / 1 + e of (-(m * b + x))

 Called the Sigmoid Equation, where e is a constant 2.718.

 Sigmoid equation produces a output between 0 and 1.

 Generalized form is: 1 / 1 + e^z, where z is our equation: -(m * b + x)
*/


// LinearRegression with gradient descent using Batches
class LogisticRegression {
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
		this.weights = tf.zeros([this.features.shape[1], 1]);
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
				this.sigmoidEqGradientDescend(
					this.features.slice([start, 0], [count, -1]),
					this.labels.slice([start, 0], [count, -1]),
				);
			}
			this.recordCrossEntropy();
			this.updateLearningRate();
		}
	}

	sigmoidEqGradientDescend(features, labels) {
		const guesses = features.matMul(this.weights).sigmoid();
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
		const guesses = this.features.matMul(this.weights).sigmoid()
		const termOne  = this.labels.transpose().matMul(guesses.log());
		const termTwo = this.labels
			.mul(tf.tensor(-1))
			.add(tf.tensor(1))
			.transpose()
			.matMul(
				guesses
					.mul(tf.tensor(-1))
					.add(tf.tensor(1))
					.log(),
			);

		const cost = termOne.add(termTwo)
			.div(this.features.shape[0])
			.mul(tf.tensor(-1))
			.get(0, 0);

		this.crossEntropyHistory.unshift(cost);
	}

	test(testFeatures, testLabels) {
		testLabels = tf.tensor(testLabels);
		const predictions = this.predict(testFeatures);
		const differences = predictions.sub(testLabels).abs();
		const incorrect = differences.sum().get();

		// divide predictions by incorrect scalar and divide by total number of predictions.
		return (predictions.shape[0] - incorrect) / predictions.shape[0];
	}

	predict(observations) {
		return this.processFeatures(observations)
			.matMul(this.weights).sigmoid()
			.greater(this.options.decisionBoundary)
			.cast('float32');
	}
}

module.exports = LogisticRegression

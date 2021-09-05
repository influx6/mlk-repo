const tf = require("@tensorflow/tfjs");
const _ = require("lodash");


class LinearRegression {

	constructor(features, labels, options) {
		this.features = features;
		this.labels = labels;
		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000,
		}, options);

		// gradient descent vars
		this.m = 0;
		this.b = 0;
	}

	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent();
		}
	}

	gradientDescent() {
		// calculation of our guess of mx + b
		// (guesses of what value could related with our
		// target label(mpg) that our feature(house power)) affects.
		const current_guesses = this.features.map(row => {
			// calculate mx + b
			const x = row[0]; // current value of horse power (our feature)
			return this.m * x + this.b;
		});

		const b_slopes = current_guesses.map((current_guess, index) => {
			return current_guess - this.labels[index][0];
		});

		const m_slopes = current_guesses.map((current_guess, index) => {
			return (this.labels[index][0] - current_guess) * (-1 * this.features[index][0]);
		});

		// b slope
		const b_mse = (_.sum(b_slopes) * 2) / this.features.length;

		// m slope
		const m_mse = (_.sum(m_slopes) * 2) / this.features.length;

		this.m = this.m - m_mse * this.options.learningRate;
		this.b = this.b - b_mse * this.options.learningRate;
	}

	test() {

	}

	predict() {

	}
}

module.exports = LinearRegression

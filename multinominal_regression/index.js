require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const _ = require('lodash');
const loadCSV = require("../data/load-csv");
const LogisticRegression = require("./lrrv3");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
	shuffle: true,
	splitTest: 50,
	converters: {
		mpg: (value) => {
			const mpg = parseFloat(value);
			if (mpg < 15) return [1, 0, 0];
			if (mpg < 30) return [0, 1, 0];
			return [0,0, 1];
		},
	},
	dataColumns: ["horsepower", "displacement",  "weight"],
	labelColumns: ["mpg"],
});

let regression = new LogisticRegression(features, _.flatMap(labels), {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 10,
	decisionBoundary: 0.5,
});

regression.train();


regression.predict([
	[215, 440, 2.16],
]).print();

const r2 = regression.test(testFeatures, _.flatMap(testLabels));

console.log(`Co-efficient of Determination: ${r2}`);


plot({
	x: regression.crossEntropyHistory.reverse(),
	xLabel: "iteration",
	yLabel: "Cross Entropy Cost"
})

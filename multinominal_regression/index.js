require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../data/load-csv");
const LogisticRegression = require("./lrrv3");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
	shuffle: true,
	splitTest: 50,
	converters: {
		passedemissions: (value) => {
			return value === "TRUE" ? 1 : 0;
		},
	},
	dataColumns: ["horsepower", "displacement",  "weight"],
	labelColumns: ["passedemissions"],
});

let regression = new LogisticRegression(features, labels, {
	learningRate: 0.5,
	iterations: 100,
	batchSize: 10,
	decisionBoundary: 0.5,
});

regression.train();

const r2 = regression.test(testFeatures, testLabels);

plot({
	x: regression.crossEntropyHistory.reverse(),
	xLabel: "iteration",
	yLabel: "Cross Entropy Cost"
})
console.log(`Co-efficient of Determination: ${r2}`);

regression.predict([
	[130, 307, 1.752],
]).print();

require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../data/load-csv");
const LinearRegression = require("./lrrv3");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("../data/cars.csv", {
	shuffle: true,
	splitTest: 50,
	dataColumns: ["horsepower", "weight", "displacement"],
	labelColumns: ["mpg"],
});

let regression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	iterations: 100,
	batchSize: 10,
});

regression.train();

const r2 = regression.test(testFeatures, testLabels);

plot({
	x: regression.mseHistory.reverse(),
	xLabel: "iteration",
	yLabel: "Mean Squared Error"
})
console.log(`Co-efficient of Determination: ${r2}`);

regression.predict([
	[120, 1.8, 350],
	[126, 0.8, 150],
]).print();

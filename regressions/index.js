require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./lrr");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
	shuffle: true,
	splitTest: 50,
	dataColumns: ["horsepower"],
	labelColumns: ["mpg"],
});

let regression = new LinearRegression(features, labels, {
	learningRate: 0.001,
	iterations: 1,
});

console.log(`Untrained LRR: M => ${regression.m} and B => ${regression.b}`);

regression.train();

console.log(`Trained LRR: M => ${regression.m} and B => ${regression.b}`);

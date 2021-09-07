require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./lrrv2");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
	shuffle: true,
	splitTest: 50,
	dataColumns: ["horsepower", "weight", "displacement"],
	labelColumns: ["mpg"],
});

let regression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	iterations: 1,
});

regression.train();

const r2 = regression.test(testFeatures, testLabels);

console.log(`Co-efficient of Determination: ${r2}`);

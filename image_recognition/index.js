require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./lrrv3");
const _ = require('lodash');
const plot = require("node-remote-plot");
const mnist = require("mnist-data");

const mnistData = mnist.training(0, 60000);
const mnistTest = mnist.testing(0, 1000);

const convertLabel = (v) => {
	const labelArr = new Array(10).fill(0);
	labelArr[v] = 1;
	return labelArr;
}

const features = mnistData.images.values.map(v => _.flatMap(v));
const testFeatures = mnistTest.images.values.map(v => _.flatMap(v));

const labels = mnistData.labels.values.map(convertLabel)
const testLabels = mnistTest.labels.values.map(convertLabel)

let regression = new LogisticRegression(features, labels, {
	learningRate: 1,
	iterations: 20,
	batchSize: 100,
});

regression.train();
test_accuracy = regression.test(testFeatures, testLabels);

console.log("TestAccuracy: ", test_accuracy, " for label: ", testLabels)
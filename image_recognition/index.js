require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./lrrv3");
const _ = require('lodash');
const plot = require("node-remote-plot");
const mnist = require("mnist-data");

const convertLabel = (v) => {
	const labelArr = new Array(10).fill(0);
	labelArr[v] = 1;
	return labelArr;
}

const [features, labels, testFeatures, testLabels] = (() => {
	let mnistData = mnist.training(0, 60000);
	let mnistTest = mnist.testing(0, 1000);

	const features = mnistData.images.values.map(v => _.flatMap(v));
	const testFeatures = mnistTest.images.values.map(v => _.flatMap(v));

	const labels = mnistData.labels.values.map(convertLabel)
	const testLabels = mnistTest.labels.values.map(convertLabel)

	return [features, labels, testFeatures, testLabels]
})()

let regression = new LogisticRegression(features, labels, {
	learningRate: 1,
	iterations: 20,
	batchSize: 100,
});

regression.train();
test_accuracy = regression.test(testFeatures, testLabels);

console.log("TestAccuracy: ", test_accuracy, " for label: ", testLabels)
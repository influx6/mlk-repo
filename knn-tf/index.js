require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require("./load-csv");



function knn(features, labels, predictionPoint, k) {
	const { mean, variance } = tf.moments(features, 0);
	const std = variance.pow(0.5); // standard deviation is sqrt(variance);
	const stdPredictionPoint = predictionPoint.sub(mean).div(std);
	return features
		.sub(mean)
		.div(std)
		.sub(stdPredictionPoint)
		.pow(2)
		.sum(1)
		.pow(0.5)
		.expandDims(1)
		.concat(labels, 1)
		.unstack()
		.sort((a, b) => {
			return a.get(0) > b.get(0) ? 1 : -1;
		})
		.slice(0, k)
		.map((b) => [b.get(0), b.get(1)])
		.reduce((acc, pair) => acc + pair[1], 0) / k;
}


let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
	shuffle: true,
	splitTest: 10,
	dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
	labelColumns: ["price"],
})

features = tf.tensor(features);
labels = tf.tensor(labels);
testFeatures = tf.tensor(testFeatures);
testLabels = tf.tensor(testLabels);

testFeatures.forEach((testPoint, index) => {
	const result = knn(features, labels, tf.tensor(testPoint), 3);
	const percentage_error = ((testLabels[index][0] - result) / testLabels[index][0]) * 100;
	console.log("Knn: ", result, percentage_error);
})

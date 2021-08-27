const outputs = [];

function distance(rowA, rowB) {
	return _.chain(rowA)
	.zip(rowB)
	.map(([a, b]) => (a - b) ** 2)
	.sum()
	.value() ** 0.5;
};

function kNNPredict(data, predictionPoint){
	return _.chain(data).map(function(row) {
		return [
			distance(_.initial(row), predictionPoint),
			row[3],
		];
	}).sortBy(row => row[0]).value();
};

function kNNGrouping(data, predictionPoint, fromK, toK) {
	const knn = kNNPredict(data, predictionPoint);
	return _.chain(knn).slice(fromK, toK).
		countBy(row => row[1]).
		toPairs().
		sortBy(row => row[1]).
		last().
		first().
		parseInt().
		value();
}

// so we are doing a: classification type of problem
// output => which bucket a ball ends in
// algorithm => k-nearest neighbor (knn) => when it backs like other dogs close to it, then it must be a dog
//
// We take a main feature, and generate enough data from it then substract from the value of feature (e.g dropPosition)
// to find out how close or far each data point is, sort them least to greatest and
// see which ones are closest to each other at the top (top 'k' records) and you will find
// the likely hood for a value for the feature that's within those ranges.

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
	outputs.push([dropPosition, bounciness, size, bucketLabel])
}

function splitDataSet(data, testCount) {
	const shuffled = _.shuffle(data);
	const testSet = _.slice(shuffled, 0, testCount);
	const trainingSet = _.slice(shuffled, testCount);
	return [testSet, trainingSet];
}

function runAnalysis() {
	const testSetSize = 50
	const [testSet, trainingSet] = splitDataSet(outputs, testSetSize);
	const accuracies = _.range(1, 20).map(k => {
		return _.chain(testSet)
		.filter((item, index) => kNNGrouping(trainingSet, item[0], 0, k) == item[3])
		.size()
		.divide(testSetSize)
		.multiply(100)
		.value();
	});
	console.log("TotalCorrect Percentage: ", accuracies);
}

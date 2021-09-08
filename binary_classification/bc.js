require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");

const readBooks = 1 // => reading books
const watchMovies = 0 // => watching movies

/* step 1. encode your choices into numeric form (binary form if two: 0 and 1)
const dataset = [
	[5, watchMovies],
	[15, watchMovies],
	[25, readBooks],
	[35, readBooks],
	[45, readBooks],
]

step 2:
 use a different equation from linear regression where we can ensure the values
 are between our encoded ranges:

 eq => 1 / 1 + e of (-(m * b + x))

 Called the Sigmoid Equation, where e is a constant 2.718.

 Sigmoid equation produces a output between 0 and 1.

 Generalized form is: 1 / 1 + e^z, where z is our equation: -(m * b + x)
*/

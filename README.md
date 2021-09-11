# ML Kits

Starter projects for learning about Machine Learning.

## Downloading

There are two ways to download this repository - either as a zip or by using git.

### Zip Download

To download this project as a zip file, find the green 'Clone or Download' button on the top right hand side. Click that button, then download this project as a zip file.

Once downloaded extract the zip file to your local computer.

### Git Download

To download this project using git, run the following command at your terminal:

```
git clone https://github.com/StephenGrider/MLKits.git
```

### Optimizations and Speed Improvements for ML with NodeJS

1. Increase allocated memory (head memory) for nodejs: `node --max-old-space-size=4096 script.js`
2. Set references not used to null when not being used anymore (use closures to make references scoped to a local scope)
3. Use tf.tidy() strategically to reduce tensorflow's memory references.
# Contributing to `cocofest`
We warmly welcome all contributions, including bug reports, bug fixes, documentation improvements, enhancements, and innovative ideas. 
But first take a look at our list of [`issues`](https://github.com/pyomeca/cocofest/issues) on GitHub.
You might find issues that already matches your needs.


## Forking `cocofest`

To make contributions, you'll need to create your own fork of the cocofest repository. Here's how to get started:

1. Visit the [cocofest project page](https://github.com/pyomeca/cocofest).
2. Click the Fork button to create your personal fork.

After forking, clone your repository to your local machine using the following command:

```bash
git clone https://github.com/your-user-name/cocofest.git
```

## Creating and activating conda environment

Before diving into development, we recommend setting up an isolated development environment, especially considering the numerous dependencies of cocofest. The quickest and most efficient way to do this is by using an Anaconda virtual environment created from the provided environment.yml file. Follow these steps:

1. Install [miniconda](https://conda.io/miniconda.html).
2. Navigate to the cocofest source directory.
3. Install the required dependencies with the following command:

```bash
conda env create -f environment.yml
```

## Implementing new features


When working on implementing new features, collaboration and communication are key to ensure a smooth development process. Here are some guidelines to follow:

- Discuss your proposed feature with the code owner to avoid conflicts with ongoing development efforts.
- Check for open pull requests to avoid duplicating work.
- If your feature is related to an existing issue, assign it to yourself. If not, open a new issue to describe your work and assign it to yourself.
- When you are ready, open a pull request with a concise yet descriptive title. Use [WIP] at the beginning of the title if it's a `work in progress`, [RTR] when it's `ready for review`, and [RTM] when it's `ready for merging`.
- Submit small, focused commits with clear and informative commit messages.
- Pay attention to feedback from maintainers and respond to comments promptly, marking them as "Done!" when resolved.
- Include a meaningful example of your new feature in the examples folder and create tests with numerical values for comparison.
- If your feature alters the API, update the ReadMe accordingly.
- Once your feature is ready for review, select Kev1CO as the reviewer on GitHub. Remove the [WIP] tag from the pull request title if necessary.
- If your pull request is accepted, congratulations! Otherwise, address the requested changes and mark comments as resolved.

## Testing your code

Adding tests is essential to merge your code into the master branch.
We strongly recommend writing tests early in the development process.
The cocofest test suite runs automatically on GitHub with every commit, but testing locally is a good practice.
To launch tests locally, run the tests folder in pytest (`pytest tests`).

## Commenting

To maintain code quality and clarity, ensure that every function, class, and module has proper docstrings following the NumPy convention.
If your new feature changes the API, update the `ReadMe.md` accordingly.

## Convention of coding

`cocofest` tries to follow as much as possible the PEP recommendations (https://www.python.org/dev/peps/). 
Unless you have good reasons to disregard them, your pull-request is required to follow these recommendations.

All variable names that could be plural should be written as such.

Black is used to enforce the code spacing. 
`cocofest` is linted with the 120-character max per line's option. 
This means that your pull-request tests on GitHub will appear to fail if black fails. 
The easiest way to make sure the black test goes through is to locally run this command:
```bash
black . -l120 --exclude "external/*"
```
If you need to install black, you can do so via conda using the conda-forge channel.
Your adherence to these conventions will help streamline the review process and enhance the overall code quality.

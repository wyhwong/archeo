# AI-Generated Test Generation

To generate tests for the `archeo` codebase,
we can use the following commands to extract the code and existing tests into separate files.
So that we can feed them into an AI model to generate new tests based on the code and existing tests.

```bash
# Extract code
find archeo -type f -name "*.py" -print0 | xargs -0 -I{} sh -c 'printf "\n\n##### %s #####\n" "{}"; cat "{}"' > codebase.txt
# Extract existing tests
find tests -type f -name "*.py" -print0 | xargs -0 -I{} sh -c 'printf "\n\n##### %s #####\n" "{}"; cat "{}"' > existing-tests.txt
```

System Prompt for AI Model:

```
You are a Senior QA Engineer with deep experience designing high-value, regression-resistant test coverage for the archeo package (simulation, inference, importance sampling, visualization, and CLI).

Your mission is to create comprehensive, actionable test cases that cover:
- Happy paths (expected behavior)
- Edge cases (boundaries, invalid inputs, unusual states)
- Negative paths (errors, exceptions, validation)
- Regression risks (areas likely to break during refactors, dependency bumps, parallelism changes, or numerical tweaks)

When writing test cases, you should prefer fast tests by default; isolate slow tests and mark them explicitly.
You should also keep the test deterministic.
Now please write out new test cases to enhance the test coverage.
```


# References
1. Segler, Marwin HS, Mike Preuss, and Mark P. Waller. "Planning chemical syntheses with deep neural networks and symbolic AI." Nature 555.7698 (2018): 604.
2. Schwaller, Philippe, et al. "“Found in Translation”: predicting outcomes of complex organic chemistry reactions using neural sequence-to-sequence models." Chemical science 9.28 (2018): 6091-6098.
3. Silver, David, et al. "Mastering the game of Go with deep neural networks and tree search." nature 529.7587 (2016): 484.
4. Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training recurrent neural networks." International Conference on Machine Learning. 2013.

# TODO

- Rollout policy network should be the same as the expansion policy network, but with a lower beam width
- Also can train a value network
    - Generate graph of reactions
    - Given a molecule A and a molecule B (fingerprints), classify if B is a parent in a path to A
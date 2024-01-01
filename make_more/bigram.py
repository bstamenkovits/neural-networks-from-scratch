import torch
import matplotlib.pyplot as plt
from typing import List


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
characters = {i:a for i, a in enumerate(['.'] + alphabet)}
indices = {a:i for i, a in enumerate(['.']+ alphabet)}


class Bigram:

    def __init__(self, model_smoothing=1, seed=None) -> None:
        self.model_smoothing = model_smoothing
        self.generator = torch.Generator().manual_seed(seed) if seed else torch.Generator()
        self.N = None
        self.P = None

    def count_bigrams(self, train_data:List[str]) -> None:
        """
        A method to count the frequency of occurance of each possible bigram 
        formed by the letters of the alphabet. These counts are stored in a 
        2-dimensional `torch.Tensor` object and is assigned to `self.N`
        """
        N = torch.zeros((27, 27), dtype=torch.int32)
        for word in train_data:
            processed_word = '.' + word + '.'
            for c1, c2 in zip(processed_word, processed_word[1:]):
                idx1 = indices[c1]
                idx2 = indices[c2]
                N[idx1, idx2] += 1  
        self.N = N

    def normalize_bigrams(self) -> None:
        """
        A method to normalize the counts of `self.N`, which is equivalent to 
        calculating the probability of occurance of each bigram. 
        """
        P = (self.N + self.model_smoothing).float()  # model smoothing: make sure no value = 0
        self.P = P / P.sum(dim=1, keepdim=True)  # normalize N over rows

    def visualize_bigrams_count(self) -> None:
        """
        Visualize the bigram counts `self.N`
        """
        plt.figure(figsize=(16,16), dpi=150)
        plt.imshow(self.N, cmap='Blues') 

        max_count = torch.max(self.N).item()

        for idx1, c1 in characters.items():
            for idx2, c2 in characters.items():
                text = c1 + c2
                count = round(self.N[idx1, idx2].item(), 2)

                color = 'black' if count/max_count < 0.5 else 'white'

                plt.text(idx2, idx1, text, ha='center', va="bottom", color=color, weight='bold')
                plt.text(idx2, idx1, count, ha='center', va="top", color=color)

        plt.axis('off')
        plt.show()

    def train(self, train_data) -> None:
        """Train the model"""
        self.count_bigrams(train_data)
        self.normalize_bigrams()

    def generate_word(self) -> str:
        """Generate a single word"""
        word, idx = [], 0
        while True:
            idx = torch.multinomial(self.P[idx], num_samples=1, replacement=True, generator=self.generator).item()
            if idx == 0: break
            word.append(characters[idx])
        return ''.join(word)

    def generate_words(self, n_words:int) -> List[any]:
        """Generate a list of words"""
        return [self.generate_word() for _ in range(n_words)]
        
    def evaluate_word(self, word:str)-> float:
        """Evaluate a single word"""
        processed_word = '.' + word + '.'
        log_likelihood, count = 0, 0
        
        for c1, c2 in zip(processed_word, processed_word[1:]):
            p = self.P[indices[c1], indices[c2]] 
            log_likelihood -= torch.log(p)  # negative log likelihood
            count += 1

        return (log_likelihood/count).item()
    
    def evaluate_words(self, words:List[str]) -> float:
        """Evaluate a list of words"""
        if self.N is None:
            return
        
        log_likelihood, count = 0, 0

        for word in words:
            log_likelihood += self.evaluate_word(word)
            count += 1
        
        return log_likelihood/count

            








+++ 
date = '2025-06-12' 
draft = false 
title = 'NLP任务指标：BLEU分数' 
categories = ['NLP任务指标'] 
tags = ['NLP任务指标', 'BLEU'] 
+++

BLEU（Bilingual Evaluation Understudy）分数用于评估机器翻译或文本生成任务中候选句子（candidate）与参考句子（reference）之间的相似度。

## 示例输入

```python
reference = "the cat is on the mat"
candidate = "the cat is on a mat"
```

## 拆分成词

- `reference_words = ["the", "cat", "is", "on", "the", "mat"]`
- `candidate_words = ["the", "cat", "is", "on", "a", "mat"]`

长度：
- `len(reference) = 6`
- `len(candidate) = 6` → 长度相等，**brevity penalty = 1**

---

## Step 1: 计算每个 n-gram 的 precision（1~4 gram）

我们分别计算 1-gram 到 4-gram 的重叠情况。

---

### n=1（Unigram）

#### Reference Unigrams:
```
["the", "cat", "is", "on", "the", "mat"]
→ Counter:
{
  'the': 2,
  'cat': 1,
  'is': 1,
  'on': 1,
  'mat': 1
}
```

#### Candidate Unigrams:
```
["the", "cat", "is", "on", "a", "mat"]
→ Counter:
{
  'the': 1,
  'cat': 1,
  'is': 1,
  'on': 1,
  'a': 1,
  'mat': 1
}
```

#### 交集（取最小值）：
```
the: min(2, 1) = 1
cat: 1
is: 1
on: 1
mat: 1
a: 0
→ overlap = 1+1+1+1+1 = 5
precision_1 = 5 / 6 ≈ 0.8333
```

---

### n=2（Bigram）

#### Reference Bigrams:
```
["the cat", "cat is", "is on", "on the", "the mat"]
```

#### Candidate Bigrams:
```
["the cat", "cat is", "is on", "on a", "a mat"]
```

#### 交集匹配：

| bigram       | in ref? | in cand? | match |
|--------------|---------|----------|-------|
| the cat      | yes     | yes      | ✅    |
| cat is       | yes     | yes      | ✅    |
| is on        | yes     | yes      | ✅    |
| on the       | yes     | no       | ❌    |
| the mat      | yes     | no       | ❌    |
| on a         | no      | yes      | ❌    |
| a mat        | no      | yes      | ❌    |

匹配项：3  
precision_2 = 3 / 5 = 0.6

---

### n=3（Trigram）

#### Reference Trigrams:
```
["the cat is", "cat is on", "is on the", "on the mat"]
```

#### Candidate Trigrams:
```
["the cat is", "cat is on", "is on a", "on a mat"]
```

#### 交集匹配：

| trigram          | in ref? | in cand? | match |
|------------------|---------|----------|-------|
| the cat is       | yes     | yes      | ✅    |
| cat is on        | yes     | yes      | ✅    |
| is on the        | yes     | no       | ❌    |
| on the mat       | yes     | no       | ❌    |
| is on a          | no      | yes      | ❌    |
| on a mat         | no      | yes      | ❌    |

匹配项：2  
precision_3 = 2 / 4 = 0.5

---

### n=4（Four-gram）

#### Reference Four-grams:
```
["the cat is on", "cat is on the", "is on the mat"]
```

#### Candidate Four-grams:
```
["the cat is on", "cat is on a", "is on a mat"]
```

#### 交集匹配：

| four-gram              | in ref? | in cand? | match |
|------------------------|---------|----------|-------|
| the cat is on          | yes     | yes      | ✅    |
| cat is on the          | yes     | no       | ❌    |
| is on the mat          | yes     | no       | ❌    |
| cat is on a            | no      | yes      | ❌    |
| is on a mat            | no      | yes      | ❌    |

匹配项：1  
precision_4 = 1 / 3 ≈ 0.3333

---

## Step 2: 精度平均

```python
precision = [0.8333, 0.6, 0.5, 0.3333]
average_precision = (0.8333 + 0.6 + 0.5 + 0.3333) / 4 ≈ 0.5667
```

---

## Step 3: brevity penalty（简洁惩罚）

候选句和参考句一样长，所以：

```python
brevity_penalty = 1
```

---

## Step 4: 最终 BLEU 分数

```python
bleu_score = brevity_penalty * average_precision * 100
           = 1 * 0.5667 * 100
           ≈ 56.67
```

---

## BLEU 分数计算流程图解

| n   | n-gram 类型 | 候选数量 | 匹配数量 | 精度     |
|-----|-------------|-----------|------------|-----------|
| 1   | unigram     | 6         | 5          | 0.8333    |
| 2   | bigram      | 5         | 3          | 0.6       |
| 3   | trigram     | 4         | 2          | 0.5       |
| 4   | four-gram   | 3         | 1          | 0.3333    |

- 平均精度：≈ 0.6232
- brevity penalty：1
- **最终 BLEU 分数：≈ 62.32**

---

## 代码实现

```python
import math
import numpy as np

from collections import Counter


def compute_bleu(reference, candidate):
    """
    Creates 1-4 grams of the reference and candidate
    and appends the precision of the candidate's to
    a list

    Args:
        reference(str): The ground truth phrase
        candidate(str): The candidate phrase

    Returns:
        bleu_score(int): bleu score across n-grams
    """
    precision = []
    reference_words = reference.split()
    candidate_words = candidate.split()
    for n in range(1, 5):
        reference_ngram = Counter(
            [
                " ".join(reference_words[i : i + n])
                for i in range(len(reference_words) + 1 - n)
            ]
        )
        candidate_ngram = Counter(
            [
                " ".join(candidate_words[i : i + n])
                for i in range(len(candidate_words) + 1 - n)
            ]
        )
        if not candidate_ngram or not reference_ngram:
            continue
        overlap = sum((reference_ngram & candidate_ngram).values())
        precision.append(overlap / sum(candidate_ngram.values()))

    brevity_penalty = (
        1
        if len(candidate) >= len(reference)
        else math.exp(1 - len(candidate) / len(reference))
    )

    print(precision)

    bleu_score = brevity_penalty * np.mean(precision) * 100
    return bleu_score


if __name__ == '__main__':
    reference = "the cat is on the mat"
    candidate = "the cat is on a mat"
    print(compute_bleu(reference, candidate))

```

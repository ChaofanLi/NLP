## 基于内部凝聚度和左右熵的新词发现
  1. 读取语料数据；
  2. n_gram统计词频，并且统计每个词左右两边词的分布，n表示预先设置的词的字数，可以设置为2到4；
  3. 计算词的凝聚度和左右熵，根据公式计算每个词的得分
    ![image](https://github.com/user-attachments/assets/79244dd5-2110-4f8a-ad2b-82f530cc12aa)
    ![image](https://github.com/user-attachments/assets/2ceaefb4-a8b3-4147-8c12-417b063ed17b)
    其中$p(w_i)$表示词w_i的出现概率
  4. 过滤标点和类似于“的”的无效词；
  5. 设置阈值，输出超过阈值的词作为新发现词。
  

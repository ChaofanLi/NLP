## 基于TF-IDF的搜索引擎
  1. 根据提供语料对文档分词，统计词频和每个词的逆文档频率；
  2. 计算每篇文档中每个词的TF-IDF，公式：
     ![image](https://github.com/user-attachments/assets/41e38f3c-c5bf-411b-8d6f-cd11f15d0568)
  4. 对输入的query进行分词，并比较在每个文档中的TF-IDF和作为得分；
  5. 排序输出得分最高的前k个文本。
  注. TF-IDF的其他用处：关键词提取、文档相似度计算、摘要生成

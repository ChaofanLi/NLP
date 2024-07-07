## 使用RNN训练一个分词模型

### 数据集构造
1. 构建词表
2. 使用jieba分词进行数据标注
3. sentence 2 sequence
4. sentence 2 label
5. padding

### 构造模型
1. embedding
2. rnn
3. linear
4. loss:CrossEntropy
5. fordward

### 模型训练
1. 模型实例化
2. 优化器实例化
3. 轮次循环
4. 梯度归零
5. 计算损失
6. 反向传播
7. 更新权重

### 保存模型
1. 保存模型参数
   
### 模型预测
1. 实例化模型
2. 加载模型参数
3. 处理输入数据
4. 预测
5. 检查输出效果

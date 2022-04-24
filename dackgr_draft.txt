Dynamic  动态
Anticipation 预期
Completion 完成
Sparse  稀疏

	• Abstract
Interpretable  能解释的，能翻译的
Dense a.稠密的
Entity  n.实体
Evidential  a.可提供证据的
Novel a.异常的
Utilize vt.利用
Latent a.潜在的
Embed  vt.栽种，使嵌入
Alleviate  vt.减轻，缓和
Datasets  数据集
State of the art 目前最高水平

	• Introduction
Query v.质疑
Dialogue  v.用对话表达
Comprehension n.理解
Incompleteness  不完全
Interpretability  可解释性
Triple   三倍
Bold arrows   粗体黑箭头
Insufficient information  不够的信息
Benchmark   基准
domain-specific  特定领域
Preliminary  初级的
Robust   鲁棒的
Inject  注入

为此，我们的预测策略把预先训练好的基于嵌入的模型的预测作为预期信息注入强化学习的状态中。

我们的完成策略动态地增加了一些额外的关系,根据当前实体的状态信息在搜索推理路径的过程中。之后，对于当前实体和一个额外的关系r，我们使用预先训练好的基于嵌入的模型来预测尾部实体e。和预测的尾部实体e将形成一个潜在的
行动（r，e），并被添加到当前实体的行动空间中。当前实体的行动空间进行路径扩展。

	• Problem formulation
	
	ε   entity set
	R   relation set
	τ   triple set
  es  head entity 
	Eo  tail   entity
	Rq  relation between  es and  eo
	
	Dout—ave  平均出度
	Threshold   阈值
	
	• Methodology
	我们首先介绍了整个多跳推理的强化学习框架，然后详细介绍了我们为稀疏KG设计的两个策略——动态预测和动态完成
	前一个策略引入了基于嵌入模型的指导信息，以帮助多跳模型在稀疏KG上找到
	在稀疏KGs上找到正确的方向
	
	动态完成策略在推理过程中引入了一些额外的行动，以增加路径的数量。
	在推理过程中引入一些额外的行动来增加路径的数量，这可以缓解KGs的稀疏性
	
	- 3.1强化学习框架
	MDP马尔科夫决策过程
	
	state：
	st = (rq, et , ht),
	Ht  historical path
	？？？LSTM是啥我没注意到的吗
	
	Action:
	action space At = {(r, e)|(et , r, e) ∈ T }
	additional action (rLOOP, et)
	???LOOP环指什么呀？
	It allows the agent to stay at the current entity
	
	Transition:
	and the transition will end at the state sT = (rq, eT , hT ).
	
	Reward:
	f(es, rq, eT ),?比较含糊
	
	- 3.2policy NETWORK
	semantic space  语义空间
	
  the action (r, e) at the step t can be represented as at = [r; e], where r and e are the vectors of r and e respectively
	
	ht = LSTM(ht−1, at−1). (1) 
	The representation of the t-th state st = (rq, et , ht) 
	can be formulated as st = [rq; et ; ht ]. 
	
	After that, we represent the action space by stacking all actions in At as At ∈ R |At|×2d , where d is the dimension of the entity and relation vector. The policy network is defined as, πθ(at |st) = σ(At(W1ReLU(W2st)))
	σ is the softmax operator, W1 and W2 are two linear neural networks, and πθ(at |st) is the probability distribution over all actions in At . 
	
	- 3.3Dynamic Anticipation
	具体来说，对于一个三重查询（es, rq, ?），我们使用预先训练好的KGE模型来获得所有实体是尾部实体的概率向量。
	所有的实体都是尾部实体
	
	p的第i个维度的值表示
	的概率，ei
	是正确的尾部实体的概率
	
	其中ep是由KGE给出的预测信息模型提供的预测信息。在本文中，我们使用以下三种策略来生成ep。
  (1) 采样策略。我们基于概率分布对一个实体进行抽样p，并将其向量表示为ep。
  (2)前一策略。我们选择在p中具有最高概率的实体p. 
  (3) 平均策略。我们根据概率分布p，取所有实体向量的加权平均值作为预测信息ep。
	在实验中，我们选择在有效集合上表现最好的策略,在有效集合上表现最好
	
	- 3.4 Dynamic Completion
  
	- 3.5Policy  Optimization   政策优化
	 the training process is obtained by maxizing the expected reward for every triplle query in the trainning set.
	
	J(θ) = E(es,r,eo)∈KGEa1,...,aT −1∈πθ[R(sT |es, r)].
	
	  θ is optimized by the Parameter  θ  of the policy network, β is the learning rate.
	
	• Experiment
	
	- 4.1   Datasets
	
	- 4.2 Experiment Setup
	
	Baseline Models
  
	For embedding-based models,coppared with
	TransE
	DisMult
	ConvE
	TuckER
  
	For multi-hop reasoning,evaluate the following models
	Neural Logical Programming(NLP)
	Neural Theorem Prover(NTP)
	MINERVA
	MultiHopKG
	
	Our model has three variations
	DacKGR (sample)
	DacKGR (top)
	DacKGR (avg)
	
	Evaluation Protocol——评估协议
	“filter"strategy   过滤策略
	We use two metrics(指标)：
	（1）MRR
	The mean reciprocal rank——平均倒数秩
	（2）K
	The proportion of correct tail entities ranking in the top K

	Implementation Details——实现细节
	
	The Dimension of the entity  and relation vectors to 200
	And using  ConvE model as the pretrained KGE for both 动态预测和动态完成策略
	Use 3-layer LSTM and set its hidden dimension to 200
	Use Adam as the optimizer
	因子 α, M and k  from  {0.5, 0.33, 0.25,0.2}, {10, 20, 40, 60} and {1, 2, 3, 5}respectively.
	Via grid search according to HITS  select the best hyperparameters (超参数)
	Reverse triple  (eo, rinvq , es) in the training set
	
- 4.3 Link Prediction Results 链接预测结果

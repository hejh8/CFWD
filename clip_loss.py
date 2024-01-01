from turtle import forward
import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F
from CLIP.clip import load

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#load clip
model, preprocess = clip.load("/home/ubuntu/Low-image/Diffusion-Low-Light-main/clip_model/ViT-B-32.pt", device=torch.device("cpu"))#"ViT-B/32"
model.to(device)
for para in model.parameters():
	para.requires_grad = False

def get_clip_score(tensor,words):
	score=0
	for i in range(tensor.shape[0]):
		#image preprocess
		clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		img_resize = transforms.Resize((224,224))
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2).unsqueeze(0)
		#get probabilitis
		text = clip.tokenize(words).to(device)
		logits_per_image, logits_per_text = model(image, text)
		probs = logits_per_image.softmax(dim=-1)
		#probs将包含经过softmax操作后的图像在不同类别上的概率分布。每个元素表示图像属于该类别的概率。
		#2-word-compared probability
		# prob = probs[0][0]/probs[0][1]#you may need to change this line for more words comparison
		if len(words)==2:
			prob = probs[0][1]/probs[0][0]##第一次反着除的
			score =score + prob

		else:
			prob = probs[0][0]
			score=score + prob

	return score


class L_clip(nn.Module):
	def __init__(self):
		super(L_clip,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, pred_LL,denoise_LL_LL):
		prompt1=["a image of well light and clear scene ","a image of low light scene "]
		prompt2=["a image of denoise and well light"]
		# prompt2=["a image of low light scene ","a image of dark scene "]
		# min_score=0
		# max_index1=-1
		# max_index2=-1
		# for i, p1 in enumerate(prompt1):
		# 	for j, p2 in enumerate(prompt2):
		# 		score = get_clip_score(denoise_LL_LL, [p1, p2])
		# 		if score < min_score:
		# 			min_score = score
		# 			max_index1 = i
		# 			max_index2 = j
		k1 = get_clip_score(pred_LL,prompt1)
		k2 = get_clip_score(denoise_LL_LL,prompt2)

		return (k1+k2)/2
	





# class L_HFRM_clip(nn.Module):
# 	def __init__(self):
# 		super(L_HFRM_clip,self).__init__()
# 		for param in self.parameters(): 
# 			param.requires_grad = False
  
# 	def forward(self,HFRM0,HFRM1):

# 		prompt=["A image with a complete and clear contour structure"]

# 		k1 = get_clip_score(HFRM0,prompt)
# 		k2 = get_clip_score(HFRM1,prompt)

# 		return k1+k2
	


# class L_HFRM_clip(nn.Module):
# 	def __init__(self):
# 		super(L_HFRM_clip,self).__init__()
# 		for param in self.parameters(): 
# 			param.requires_grad = False
  
# 	def forward(self,HFRM0):

# 		prompt=["A image with a complete and clear contour structure"]


# 		k1 = get_clip_score(HFRM0,prompt)


# 		return k1
	


class Prompts(nn.Module):
	def __init__(self,initials=None):
		super(Prompts,self).__init__()
		if initials!=None:
			text = clip.tokenize(initials).cuda()
			with torch.no_grad():
				self.text_features = model.encode_text(text).cuda()
		else:
			self.text_features=torch.nn.init.xavier_normal_(nn.Parameter(torch.cuda.FloatTensor(2,512))).cuda()

	def forward(self,tensor):
		for i in range(tensor.shape[0]):
			image_features=tensor[i]
			nor=torch.norm(self.text_features,dim=-1, keepdim=True)
			similarity = (model.logit_scale.exp() * image_features @ (self.text_features/nor).T).softmax(dim=-1)
			if(i==0):
				probs=similarity
			else:
				probs=torch.cat([probs,similarity],dim=0)
		return probs
	#在 forward 方法中，接受一个张量 tensor 作为输入。然后通过一个循环遍历张量的第一维（假设张量是一个批量的图像特征）。
	#对于每个图像特征，计算其与 text_features 的相似度。具体来说，首先计算 text_features 的 L2 范数，并将其保持为一个列向量。
	#然后计算图像特征与标准化后的 text_features 的点积，再经过缩放和 softmax 操作得到相似度。最后将每次计算得到的相似度拼接起来形成最终的结果 probs，并将其返回。

learn_prompt=Prompts().cuda()
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))

def get_clip_score_from_feature(tensor,text_features):
	score=0
	for i in range(tensor.shape[0]):
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2.reshape(1,3,224,224))
  
		image_features = model.encode_image(image)
		image_nor=image_features.norm(dim=-1, keepdim=True)
		nor= text_features.norm(dim=-1, keepdim=True)
		#接下来，使用 CLIP 模型的 encode_image 方法对图像进行编码，得到图像特征 image_features。
		#然后分别计算图像特征和文本特征的 L2 范数，并将其保持为列向量。接着计算图像特征与文本特征的点积，并进行缩放和 softmax 操作，得到相似度 similarity
		similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1)
		probs = similarity
		prob = probs[0][0]
		score =score + prob
	score=score/tensor.shape[0]
	return score
#该函数的作用是根据输入的图像特征张量和文本特征向量，计算图像与文本之间的相似度得分，并返回平均得分


class L_clip_from_feature(nn.Module):
	def __init__(self):
		super(L_clip_from_feature,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x, text_features):
		k1 = get_clip_score_from_feature(x,text_features)
		return k1

#for clip reconstruction loss
res_model, res_preprocess = load("/home/ubuntu/Low-image/Diffusion-Low-Light-main/clip_model/RN101.pt", device=device)
for para in res_model.parameters():
	para.requires_grad = False


#pred_conv_features 表示预测的特征列表，input_conv_features 表示输入的特征列表，weight 表示权重。
def l2_layers(pred_conv_features, input_conv_features,weight):
	weight=torch.tensor(weight).type(pred_conv_features[0].dtype)
	return weight@torch.tensor([torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
			zip(pred_conv_features, input_conv_features)],requires_grad=True)/len(weight)
#在函数中，首先通过 torch.tensor() 将权重转换为与 pred_conv_features 的数据类型相同的张量。
#然后，使用列表推导式和 zip() 函数，遍历 pred_conv_features 和 input_conv_features 中的特征，并计算每对特征之间的平方差的均值
#接着，将计算得到的平方差均值列表转换为张量，并使用 @ 运算符进行矩阵乘法操作，将权重与平方差均值进行加权求和。最后，将加权求和结果除以权重的长度，得到加权平均值。

def get_clip_score_MSE(pred,inp,weight):
	score=0
	for i in range(pred.shape[0]):

		pred_img=img_resize(pred[i])
		
		pred_img=pred_img.unsqueeze(0)
	
		pred_img=clip_normalizer(pred_img.reshape(1,3,224,224))
		pred_image_features = res_model.encode_image(pred_img)

		inp_img=img_resize(inp[i])
		inp_img=inp_img.unsqueeze(0)
		inp_img=clip_normalizer(inp_img.reshape(1,3,224,224))
		inp_image_features = res_model.encode_image(inp_img)
		
		MSE_loss_per_img=0
		for feature_index in range(len(weight)):
				MSE_loss_per_img=MSE_loss_per_img+weight[feature_index]*F.mse_loss(pred_image_features[1][feature_index].squeeze(0),inp_image_features[1][feature_index].squeeze(0))
		score = score + MSE_loss_per_img
	return score
#该函数的作用是根据输入的预测图像张量、输入图像张量和特征权重，计算预测图像和输入图像之间的均方误差得分，并返回总的得分。

class L_clip_MSE(nn.Module):
	def __init__(self):
		super(L_clip_MSE,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
		
	def forward(self, pred, inp,weight=[1.0,1.0,1.0,1.0,0.5]):
		res = get_clip_score_MSE(pred,inp,weight)
		return res



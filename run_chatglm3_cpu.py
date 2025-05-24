from modelscope import AutoTokenizer, AutoModel
model_dir = "./chatglm3-6b"  
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float() 
model = model.eval()
response, history = model.chat(tokenizer, "请说出以下两句话区别在哪⾥？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少", history=[])
print(response)

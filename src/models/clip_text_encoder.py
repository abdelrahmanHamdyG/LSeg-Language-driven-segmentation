import torch 
import torch.nn as nn
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"


class ClipTextEncoder(nn.Module):
    def __init__(self, model ): 
        super(ClipTextEncoder, self).__init__()
        
        self.model = model
        self.model.eval() 


    def forward(self, text_prompts): 
    
        
        tokens = clip.tokenize(text_prompts).to(device) # tokens [N,77] 
        

        with torch.no_grad():
            text_features = self.model.encode_text(tokens) 
        
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [N,512]
        


        return text_features
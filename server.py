import fastapi
import torch
import torchvision.transforms as transforms
import random
import model
from model import DDPM
import io
import base64
from torchvision.utils import save_image

app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
    
  n_T = 500 # 500
  device=torch.device("cpu")
  n_classes = 20
  n_feat = 128 # 128 ok, 256 better (but slower)

  app.state.digit = DDPM(nn_model=model.ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=torch.device("cpu"), drop_prob=0.1)
  app.state.digit.load_state_dict(torch.load("new_model_9.pth", map_location='cpu'))
  app.state.digit.eval()


@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }

@app.post("/mnist_generator")
async def draw_digits(request: fastapi.Request):
  text = (await request.json())["text"]
  print("Input text:", text)
  conditions = text.split()
  context_label = int(conditions[0]) + int(conditions[1]) *  int(conditions[0])
  context_label = torch.tensor(context_label)
  with torch.no_grad():
    x_gen, x_gen_store = app.state.digit.sample_single(context_label)
    
    buffer = io.BytesIO()
    save_image(x_gen[0], buffer, format='PNG')
    #img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    img_str = f"data:image/png;base64,{img_str}"
    return { "img": img_str }

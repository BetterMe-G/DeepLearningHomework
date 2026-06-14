# Avatar Studio

文本驱动的高保真虚拟形象生成,基于 **StyleGAN2-FFHQ**,无 diffusion。
单卡推理 ~30–100 ms / 1024×1024。社交平台虚拟形象场景已内置 10 个 Mapper 预设。

---

## 文本编辑方式

| 方式 | 是否要训 | 训练粒度 | 一次训练能用多少 prompt |
|---|---|---|---|
| **StyleCLIP Mapper** | ✅ | 1 prompt ≈ 1h(3090) | **一句** |

- 一句 prompt 想要最稳最像,选 **Mapper**(本仓库主要工作流)。

---

## 0. 前置准备

### 0.1 安装环境

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 0.2 下载权重

文件名必须严格匹配下面给出的名字,放到 `checkpoints/` 下。直接复制粘贴下面这段一次性跑完即可:

```bash
pip install gdown
mkdir -p checkpoints && cd checkpoints

# 1) StyleGAN2-FFHQ (rosinality port, 380 MB)
gdown 1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK -O stylegan2-ffhq-config-f.pt

# 2) e4e FFHQ encoder (1.1 GB) —— encoder4editing 官方 release
gdown 1cUv_reLE6k3604or78EranS7XzuVMWeO -O e4e_ffhq_encode.pt

# 3) ArcFace IR-SE-50 (175 MB) —— TreB1eN/InsightFace_Pytorch
gdown 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn -O model_ir_se50.pth

# 4) dlib 68 landmarks (100 MB,bz2 压缩,需解压)
wget -O shape_predictor_68_face_landmarks.dat.bz2 \
     http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 -f shape_predictor_68_face_landmarks.dat.bz2

```
---

## 2. 批量训 10 个 Mapper(~10h,3090)

10 个社交平台向预设(见 `scripts/train_all_mappers.sh`):

| name | prompt |
|---|---|
| holographic | a person with iridescent holographic hair |
| cyber_tattoo | a person with cyberpunk neon face tattoos |
| golden_freckles | a person with golden freckles constellation across the face |
| pastel_split | a person with pastel pink and platinum split dyed hair |
| dark_academia | a person with dark academia aesthetic and vintage round glasses |
| egirl_heart | a person with e-girl aesthetic, heart shaped cheek blush |
| baroque_oil | a baroque oil painting portrait of a person |
| kabuki | a person with kabuki theater white face makeup and red accent lines |
| crystal_skin | a person with crystals growing from the skin |
| neon_noir | a person in dramatic neon noir lighting, pink and blue rim light |

启动:

```bash
bash scripts/train_all_mappers.sh                # 训所有还没训过的
bash scripts/train_all_mappers.sh holographic    # 只训一个
ITERATIONS=20000 bash scripts/train_all_mappers.sh   # 快速试跑(质量略低)
```

特性:

- **可断点恢复**:已存在 `checkpoints/mappers/<name>.pt` 的会自动跳过。
- **每个 prompt 一份日志**:`logs/mappers/<name>.log`。
- **训练过程定期出图**:`checkpoints/mappers/<name>_samples/iter_xxxxxx_{src,edit}.png`,可在训练中肉眼看效果是否在收敛。

单个 prompt 的内部循环(`avatar_studio/edit/mapper.py:MapperTrainer.step`):

```
for it in 1..50000:
    w  = G.sample_w(batch=2, truncation=0.7)
    Δw = mapper(w) * 0.1
    img_src  = G(w);  img_edit = G(w + Δw)
    loss = λ_clip·CLIP(img_edit, prompt)
         + λ_id  ·(1 - cos(arcface(img_edit), arcface(img_src)))
         + λ_l2  ·‖Δw‖²
    loss.backward(); opt.step()
```

只有 mapper 的 ~10M 参数在更新,SG2 / CLIP / ArcFace 全程冻结。

---

## 3. 视觉验收(~5 分钟)

对所有 ckpt 用同一组随机种子出 before/after 对照网格,挑掉效果差的:

```bash
python scripts/eval_mappers.py --out_dir eval_grids/ --n 8 --strength 0.1
```

`eval_grids/<name>.png` 每张 2 行 × 8 列:上排原图,下排编辑后。

判断标准:

- **通过**:文本特征明显出现,人脸结构没破,8 个种子里至少 6 个一致。
- **重训**:特征没出现 / 出现但破脸 / 强度不一致。把对应 `.pt` 删掉,重跑 `train_all_mappers.sh <name>` 即可(可调大 `--lambda_clip` 或加 `ITERATIONS`)。

---

## 4. 上线检查

在 `AvatarPipeline` 里按名字调用任一 mapper:

```python
from avatar_studio.pipeline import AvatarPipeline
pipe = AvatarPipeline.from_config()
img = pipe.generate(
    text="holographic",                # 或直接 prompt 文本
    mapper_ckpt="checkpoints/mappers/holographic.pt",
    seed=42,
).image
img.save("out.png")
```

---

## 端到端流程图

```
text ───────────→ Mapper(text) ─→ Δw+

photo (opt) ─→ align ─→ e4e ─→ W+ pivot
                                 │
          W+(随机 or pivot) + Δw+ ───────────→ image
```

---

## 项目结构

```
avatar_studio/
├── README.md
├── requirements.txt
├── configs/default.yaml
├── avatar_studio/
│   ├── config.py
│   ├── pipeline.py              # AvatarPipeline 顶层入口
│   ├── models/{stylegan2.py, e4e.py, clip_loss.py, id_loss.py}
│   ├── edit/{mapper.py}
│   ├── vendor/                  # 第三方代码,已内嵌
│   │   ├── stylegan2/{model.py, op/}
│   │   ├── encoders/{helpers.py, psp_encoders.py}
│   │   ├── psp.py
│   │   └── insightface_model.py
│   └── utils/{image.py, logger.py}
└── scripts/
    ├── train_mapper.py          # 训单个 mapper
    ├── train_all_mappers.sh     # 批量训 10 个预设(可断点恢复)
    ├── eval_mappers.py          # 出 before/after 网格,人工验收
    ├── invert.py                # e4e 反演
    └── generate.py              # 一键生成
```

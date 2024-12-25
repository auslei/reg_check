#%%
import pymupdf4llm

path = "./data/ncc2022-volume-two-20230501b.pdf"
file = pymupdf4llm.to_markdown(path, write_images=True, image_path="images")
#%%
with open("output.md", "w") as f:
    f.write(file)
# %%

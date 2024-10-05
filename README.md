このコードは LamaRemover (今の名前は iopaint です) を極力シンプルに Python コード内部から呼び出すことが出来るようにしたものです。

標準の Lama Remover (iopaint) では、Web UI の利用が前提となっているようです。これはプログラムで自動処理したい時に少々手間です。

----

This code is designed to allow calling LamaRemover (now known as iopaint) from within Python code as simply as possible.

The standard Lama Remover (iopaint) seems to assume the use of a Web UI. This can be somewhat inconvenient when you want to perform automated processing programmatically.

----

```python
from PIL import Image
from lamarem import LaMaRemover

image_path = "image1.png"
mask_path = "image2.png"

remover = LaMaRemover()
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

output_image = remover(image, mask)
output_image.save("output.png")
```

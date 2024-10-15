import torch
from torchvision import transforms as T
import gradio as gr

class App:

    title = 'Scene Text Recognition with<br/>Permuted Autoregressive Sequence Models'
    models = ['parseq', 'parseq_tiny', 'abinet', 'crnn', 'trba', 'vitstr']

    def __init__(self):
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def _get_model(self, name):
        if name in self._model_cache:
            return self._model_cache[name]
        model = torch.hub.load('baudm/parseq', name, pretrained=True).eval()
        self._model_cache[name] = model
        return model

    @torch.inference_mode()
    def __call__(self, model_name, image):
        if image is None:
            return '', []
        model = self._get_model(model_name)
        image = self._preprocess(image.convert('RGB')).unsqueeze(0)
        # Greedy decoding
        pred = model(image).softmax(-1)
        label, _ = model.tokenizer.decode(pred)
        raw_label, raw_confidence = model.tokenizer.decode(pred, raw=True)
        # Format confidence values
        max_len = 25 if model_name == 'crnn' else len(label[0]) + 1
        conf = list(map('{:0.1f}'.format, raw_confidence[0][:max_len].tolist()))
        return label[0], [raw_label[0][:max_len], conf]


def main():
    app = App()

    with gr.Blocks(analytics_enabled=False, title=app.title.replace('<br/>', ' ')) as demo:
        model_name = gr.Radio(app.models, value=app.models[0], label='The STR model to use')
        with gr.Tabs():
            with gr.TabItem('Image Upload'):
                image_upload = gr.Image(type='pil', label='Image')
                read_upload = gr.Button('Read Text')

        output = gr.Textbox(max_lines=1, label='Model output')
        #adv_output = gr.Checkbox(label='Show detailed output')
        raw_output = gr.Dataframe(row_count=2, col_count=0, label='Raw output with confidence values ([0, 1] interval; [B] - BLANK token; [E] - EOS token)')

        read_upload.click(app, inputs=[model_name, image_upload], outputs=[output, raw_output])
        #adv_output.change(lambda x: gr.update(visible=x), inputs=adv_output, outputs=raw_output)
    demo.queue(max_size=20)
    demo.launch()


if __name__ == '__main__':
    main()
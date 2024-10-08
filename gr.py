import gradio as gr
from flux_pipeline import FluxDevPipeline, FluxSchnellPipeline, seed_torch, builder
import os


class ModelManager():
    def __init__(self):
        self.current_model_name = None
        self.pipe = None
        self.builder = builder

    def pre_check(self, model_select="sd3", check_type=None):
        full_model_path = []
        check_pass = True
        model_select_path = os.path.join('models')
        clip_path = os.path.join(model_select_path, "clip_combined.bmodel")
        dev_path = os.path.join(model_select_path, "dev_w4bf16_transform_combined.bmodel")
        schnell_path = os.path.join(model_select_path, "schnell_w4bf16_transform_combined.bmodel")
        t5_path = os.path.join(model_select_path, "t5_combined.bmodel")
        vae_de_path = os.path.join(model_select_path, "vae_decoder_F16.bmodel")

        if "clip_path" in check_type:
            if not os.path.isfile(clip_path):
                gr.Warning("No {}, please download first".format(clip_path))
                check_pass = False
                # return False
        if "dev" in check_type:
            if not os.path.isfile(dev_path):
                gr.Warning("No {}, please download first".format(dev_path))
                check_pass = False

        if "schnell" in check_type:
            if not os.path.exists(schnell_path):
                gr.Warning("No {}, please download first".format(schnell_path))
                check_pass = False
        if "t5" in check_type:
            if not os.path.exists(t5_path):
                gr.Warning("No {} t5, please download first".format(t5_path))
                check_pass = False

        if "vae" in check_type:
            if not os.path.exists(vae_de_path):
                gr.Warning("No {} vae, please download first".format(vae_de_path))
                check_pass = False

        full_model_path.append(clip_path)
        full_model_path.append(dev_path)
        full_model_path.append(schnell_path)
        full_model_path.append(t5_path)
        full_model_path.append(vae_de_path)

        return check_pass, full_model_path

    def change_model(self, model_select, progress=gr.Progress()):
        if model_select == []:
            model_select = None
        if model_select is not None:
            if self.pipe is None or self.current_model_name != model_select:
                check_pass, full_model_path = self.pre_check(check_type=["clip_path", model_select, "t5", "vae"])
                del self.pipe
                if check_pass:
                    if model_select == "schnell":
                        self.pipe = FluxSchnellPipeline(
                                                self.builder,
                                                full_model_path[0],
                                                full_model_path[3],
                                                full_model_path[2],
                                                full_model_path[4],
                                   "./models/tokenizer",
                                   "./models/tokenizer_2",
                                   "./models/ids_emb.pt",
                                                0
                                                )
                        self.current_model_name = model_select
                        gr.Info("{} load success".format(model_select))

                        return self.current_model_name

                    elif model_select == "dev":
                        self.pipe = FluxDevPipeline(
                                                self.builder,
                                                full_model_path[0],
                                                full_model_path[3],
                                                full_model_path[1],
                                                full_model_path[4],
                                   "./models/tokenizer",
                                   "./models/tokenizer_2",
                                   "./models/ids_emb.pt",
                                                0
                                                )
                        self.current_model_name = model_select
                        gr.Info("{} load success".format(model_select))

                        return self.current_model_name

                else:
                    gr.Error("{} models are not complete".format(model_select))

        else:
            gr.Info("Please select a model")
            return None

    def generate_image_from_text(self,
                                 input_prompt_1,
                                 input_prompt_2,
                                 steps=20,
                                 guidance_scale=7.0,
                                 seed_number=0
                                 ):
        if self.pipe is None:
            gr.Info("Please select a model")
            return None

        else:
            seed_torch(seed_number)
            if input_prompt_1 == "":
                gr.Warning("please input your prompt")
                return None

            if input_prompt_2 == "":
                input_prompt_2 = None

            img_pil = self.pipe(input_prompt_1,
                                input_prompt_2,
                                guidance_scale,
                                steps
                                )

            return img_pil


    def update_slider(self, model_select):
        if model_select == "dev":
            return 28
        elif model_select == "schnell":
            return 4



model_manager = ModelManager()

description = """
# FLUX.1 on Airbox 🥳
"""

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_prompt_1 = gr.Textbox(lines=1, label="Prompt", value=None)

                with gr.Row():
                    input_prompt_2 = gr.Textbox(lines=1, label="T5 Prompt(Optional)", value=None)

                with gr.Row():
                    num_step = gr.Slider(minimum=1, maximum=50, value=4, step=1, label="Steps", scale=2)
                    guidance_scale = gr.Slider(minimum=0, maximum=20, value=0, step=0.1, label="CFG scale", scale=2)

                with gr.Row():
                    seed_number = gr.Number(value=-1, label="Seed", scale=1, minimum=-1, info="-1 is random")


                with gr.Row():
                    clear_bt = gr.ClearButton(value="Clear",
                                              components=[input_prompt_1,
                                                          input_prompt_2,
                                                          num_step,
                                                          guidance_scale,
                                                          ]
                                              )
                    submit_bt = gr.Button(value="Submit", variant="primary")

            with gr.Column():
                with gr.Row():
                    model_select = gr.Dropdown(choices=["schnell", "dev"], value=None, label="Model", interactive=True)
                    load_bt = gr.Button(value="Load Model", interactive=True)
                out_img = gr.Image(label="Output", format="png")

        with gr.Row():
            with gr.Column():
                example = gr.Examples(
                    label="Example",
                    examples=[
                        ["high detailed texture, photograph, realistic, RAW photo of young lady, emo, dark hair, makeup, depth of field, soft split ambient window light, soft realistic shadows, perfect composition",
                         "",
                         4,
                         0.0,
                         -1,
                         "schnell"],

                        [
                        "Beautiful, stylish and sexy Korean girl in a sexy sheer bikini with a smile on her face.",
                        "",
                        28,
                        0.0,
                        -1,
                        "dev"
                        ],
                    ],
                    inputs=[input_prompt_1, input_prompt_2, num_step, guidance_scale, seed_number, model_select]
                )

        clear_bt.add(components=[out_img])
        model_select.change(model_manager.update_slider, model_select, num_step)

        load_bt.click(model_manager.change_model, [model_select], [model_select])

        input_prompt_1.submit(model_manager.generate_image_from_text,
                              [input_prompt_1,
                               input_prompt_2,
                               num_step,
                               guidance_scale,
                               seed_number],
                              [out_img]
                              )
        input_prompt_2.submit(model_manager.generate_image_from_text,
                              [input_prompt_1,
                               input_prompt_2,
                               num_step,
                               guidance_scale,
                               seed_number],
                              [out_img]
                              )
        submit_bt.click(model_manager.generate_image_from_text,
                        [input_prompt_1,
                         input_prompt_2,
                         num_step,
                         guidance_scale,
                         seed_number],
                        [out_img]
                        )

    # 运行 Gradio 应用
    demo.queue(max_size=10)
    demo.launch(server_port=8999, server_name="0.0.0.0")
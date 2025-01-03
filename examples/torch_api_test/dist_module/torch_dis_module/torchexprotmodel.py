import torch
from torch.export import export

class AutoExport:
    def __init__(self, strict=True):
        self.strict = strict
    def __call__(self, model, *args, **kwargs):
        try:
            return export(model, args=args, kwargs=kwargs, strict=self.strict)
        except Exception as e:
            print(f"Error during export: {e}")
            return None

# import torch
# from torchvision.models import vit_b_16  # Import Vision Transformer model

# # Initialize the ViT model
# model = vit_b_16(pretrained=False)  # Set pretrained=True for a pre-trained model

# dummy_input= torch.randn(1, 3, 224, 224)

# exporter = AutoExport(strict=True)
# exported_program = exporter(model, dummy_input)


# print(exported_program)
# if __name__ == '__main__':
#     class DummyModel(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear = torch.nn.Linear(10, 5)

#         def forward(self, x):
#             return self.linear(x).relu()

#     model = DummyModel()
#     dummy_input = torch.randn(1, 10)

#     exporter = AutoExport()
#     exported_program = exporter(model, dummy_input)
#     if exported_program:
#         print(exported_program)

#     exporter_non_strict = AutoExport(strict=False)
#     exported_program_non_strict = exporter_non_strict(model, dummy_input)
#     if exported_program_non_strict:
#         print(exported_program_non_strict)

#     def dummy_function(x, y):
#         return torch.sin(x) + torch.cos(y)
    
#     exported_function = exporter(dummy_function, torch.randn(3,3), torch.randn(3,3) )
#     if exported_function:
#        print(exported_function)


#     class ContextManager():
#         def __init__(self):
#             self.count = 0
#         def __enter__(self):
#             self.count += 1
#         def __exit__(self, exc_type, exc_value, traceback):
#             self.count -= 1

#     class ModelWithContext(torch.nn.Module):
#         def forward(self, x):
#             with ContextManager():
#                 return x.sin() + x.cos()

#     model_with_context = ModelWithContext()
    
#     exported_with_context_strict = exporter(model_with_context, torch.ones(3,3))
#     if exported_with_context_strict:
#         print("context model strict export passed which is unexpected")
#     else:
#       print("context model strict export failed whcih is expected")

#     exported_with_context_non_strict = exporter_non_strict(model_with_context, torch.ones(3,3))
#     if exported_with_context_non_strict:
#         print("context model non strict export success")
#         print(exported_with_context_non_strict )
#     else:
#       print("context model non strict export failed which is unexpected")
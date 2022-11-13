import torch
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("model1", type=str)
        parser.add_argument("model2", type=str)
        parser.add_argument("output_path", type=str)
        parser.add_argument("--weight","-w", type=float, default=0.99)

        args = parser.parse_args()

        teacher_model = torch.load(args.model1, 'cpu')
        s_model = torch.load(args.model2, 'cpu')
        if 'model' in teacher_model:
                teacher_model = teacher_model['model']
        for key in teacher_model['state']:
                s_model['model']['state'][key] = args.weight*teacher_model['state'][key] + (1-args.weight)*s_model['model']['state'][key]

        torch.save(s_model, args.output_path)

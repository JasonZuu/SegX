python interpret.py --dataset ham --model resnet --algo grad_cam
python interpret.py --dataset ham --model densenet --algo grad_cam
python interpret.py --dataset chestx --model resnet --algo grad_cam
python interpret.py --dataset chestx --model densenet --algo grad_cam

python interpret.py --dataset ham --model resnet --algo segx_gradcam
python interpret.py --dataset ham --model densenet --algo segx_gradcam
python interpret.py --dataset chestx --model resnet --algo segx_gradcam
python interpret.py --dataset chestx --model densenet --algo segx_gradcam

python interpret.py --dataset ham --model resnet --algo gradient_shap
python interpret.py --dataset ham --model densenet --algo gradient_shap
python interpret.py --dataset chestx --model resnet --algo gradient_shap
python interpret.py --dataset chestx --model densenet --algo gradient_shap

python interpret.py --dataset ham --model resnet --algo segx_gradient_shap
python interpret.py --dataset ham --model densenet --algo segx_gradient_shap
python interpret.py --dataset chestx --model resnet --algo segx_gradient_shap
python interpret.py --dataset chestx --model densenet --algo segx_gradient_shap

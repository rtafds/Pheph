#pip install wheel twine 

python setup.py bdist_wheel
twine upload --repository testpypi dist/*

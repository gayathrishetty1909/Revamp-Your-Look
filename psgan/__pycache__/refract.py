from refract import Refract

class PostProcess:
    def __init__(self, device: str, config):
        # Your initialization code here

     def __call__(self, source: Image, result: Image):
        # Your preprocessing code here

        # Call the Refract function
        source, result = Refract(source, result)

        # Your post-processing code here
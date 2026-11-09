__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void median_kernel(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
 
	uint otoczenie [9];

	int index = 0;
	for (int i = coord.x - 1; i <= coord.x + 1; i++)
	{
		for (int j = coord.y - 1; j <= coord.y + 1; j++)
		{
			uint4 pixel  = read_imageui(inputImage, imageSampler, (int2)(i, j));
			uint pixel_grey = pixel.y;

			otoczenie[index] = pixel_grey;
			index++;
		}
	}

	// sortowanie bąbelkowe otoczenia
	for (int i = 0; i < 9 - 1; i++)
	{
		for (int j = 0; j < 9 - i - 1; j++)
		{
			if (otoczenie[j] > otoczenie[j + 1])
			{
				uint temp = otoczenie[j];
				otoczenie[j] = otoczenie[j + 1];
				otoczenie[j + 1] = temp;
			}
		}
	}	

	uint mediana = otoczenie[4]; // wartość mediany
	write_imageui(outputImage, coord, (uint4)((uint)(mediana), (uint)(mediana), (uint)(mediana), 255)); 		
}
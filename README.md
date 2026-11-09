### Użycie

```sh
python main.py --path highway
python main.py --source 0
python main.py --path highway --opencl-path ./opencl_pipeline.py
```

Te same ramki BRG wpadają na dwa "Pipeline'y":

- OpenCV reference: 3×3 median -> MOG2 mask.
- OpenCL: `Pipeline.process(frame)` -> output image.

`--opencl-path` wybiera plik z klasą `Pipeline` implementującą metodę
`process(frame)`.
Metoda ta otrzymuję aktualną ramkę i zwraca ramkę wyjściową.

Aktualnie wspierana jest tylko testowa klasa wykonująca medianę.

### Literatura

- Bazowy artykuł dotyczący MOG - [Adaptive background mixture models for real-time tracking](https://doi.org/10.1109/CVPR.1999.784637)
- Potem jego rozwinięcie - [Learning patterns of activity using real-time tracking](https://doi.org/10.1109/34.868677)
- Wydajna implementacja per pixel - [Efficient adaptive density estimation per image pixel for the task of background subtraction](https://doi.org/10.1016/j.patrec.2005.11.005)
- Rewiew różnych metod MOG - [Review of background subtraction methods using Gaussian mixture model for video surveillance systems](https://doi.org/10.1007/s10462-017-9542-x)
- Optymalizacja na intach - [An optimisation of Gaussian mixture models for integer processing units](https://doi.org/10.1007/s11554-014-0402-5)
- Implementacja na GPU - [Fast background modeling using GMM on GPU](https://doi.org/10.1109/ICALIP.2014.7009932)

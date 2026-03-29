### About the app

This app aims to demonstrate the most cost-effective power sources for an islanded data centre. Building it also served as something interesting to busy myself with while outdoorsy hobbies were hampered by a broken rib and dreary Midlands weather.

The idea was inspired by Izzy Woolgar and Ryan Jenkinson's article, [How to accelerate the UK’s AI revolution](https://microgridai.centrefornetzero.org/).

#### How to use it

1. Select a PV and a Wind location. If you want. If you don't, that's fine too - it'll default to using the centroid of the UK.
2. Select generation, storage, and carbon price options. These are used as inputs for the optimiser.
3. Run the optimiser. It may take up to 45s if you give it something difficult.
4. Inspect the results, test out some other cost scenarios, have fun, or try to break it.

#### How it works

I've written about this in my [GitHub pages](https://elisjackson.github.io/).

#### Future to-dos

- Tidy up repo & make public
- Include discounting
- Add an option for electricity from the grid?
- Use a "typical meteorological year" weather file - the current weather data (ERA5) uses 2025 weather. This was an unusually sunny year in the UK, which likely makes Solar PV come out a bit more favourable than it should
- Any other ideas? Let me know through [my LinkedIn](https://www.linkedin.com/in/elis-jackson-a428801a5)

#### Data sources

- **Weather data**: ERA5 (2025 weather)
- **Generation and storage assumptions**: Various sources, noted in the Optimiser Parameters section. [GNESTE](https://github.com/iain-staffell/GNESTE) helped me to source some of these.
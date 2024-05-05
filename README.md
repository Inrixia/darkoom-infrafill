# Darkroom Infrafill

This is a fork of **Signynt's** amazing **[Darkroom Script](https://github.com/Signynt/signynts-darkroom-script)** talored specifically for removing dust & scratches from film negatives using a HDRi tiff containing infared and negative scan pages.

This currently is only setup to work with Silverfast HDRi scans. But can be easily modified to work with anything else by setting the `ir` and `img` values properly.

## Installation

1. Install [Python](https://www.python.org/downloads/)
2. [Download](https://github.com/Inrixia/darkoom-infrafill/archive/refs/heads/main.zip) or clone this repository
3. Run initial setup `pip install -r requirements.txt`

## Usage

Run the script against the folder/file you want to process with a given output directory:

```bash
python darkroom-infrafill.py ./input ./output
```

or

```bash
python darkroom-infrafill.py ./myrawscan.tif ./output
```

Output files will have a transparency layer containing the mask from the infrared scan which you can use to do infill in photoshop or a app of your choice.

#!/bin/bash
# Mirror script to push to both repositories

echo "Pushing to ElBalor repository..."
git push origin main

echo "Pushing to De-ArchAngel7 repository..."
git push mirror main

echo "Mirroring complete!"

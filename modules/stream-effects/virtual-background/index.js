// @flow

import JitsiStreamBackgroundEffect from './JitsiStreamBackgroundEffect';
import createTFLiteModule from './vendor/tflite/tflite';
import createTFLiteSIMDModule from './vendor/tflite/tflite-simd';
const models = {
    modelLandscape: 'libs/selfie_segmentation_landscape.tflite'
};

let tflite;
let wasmCheck;
let isWasmDisabled = false;

const segmentationDimensions = {
    modelLandscape: {
        height: 144,
        width: 256
    }
};

/**
 * Creating a wrapper for promises on a specific time interval.
 * 
 * https://github.com/jitsi/jitsi-meet
 * jitsi-meet/react/features/virtual-background/functions.js
 *
 * @param {number} milliseconds - The number of milliseconds to wait the specified
 * {@code promise} to settle before automatically rejecting the returned
 * {@code Promise}.
 * @param {Promise} promise - The {@code Promise} for which automatic rejecting
 * after the specified timeout is to be implemented.
 * @returns {Promise}
 */
 export function timeout(milliseconds: number, promise: Promise<*>): Promise<Object> {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            reject(new Error('408'));

            return;
        }, milliseconds);

        promise.then(resolve, reject);
    });
}

/**
 * Creates a new instance of JitsiStreamBackgroundEffect. This loads the Meet background model that is used to
 * extract person segmentation.
 *
 * @param {Object} virtualBackground - The virtual object that contains the background image source and
 * the isVirtualBackground flag that indicates if virtual image is activated.
 * @returns {Promise<JitsiStreamBackgroundEffect>}
 */
export async function createVirtualBackgroundEffect(virtualBackground: Object) {
    if (!MediaStreamTrack.prototype.getSettings && !MediaStreamTrack.prototype.getConstraints) {
        throw new Error('JitsiStreamBackgroundEffect not supported!');
    }

    // Checks if WebAssembly feature is supported or enabled by/in the browser.
    // Conditional import of wasm-check package is done to prevent
    // the browser from crashing when the user opens the app.

    if (!tflite && !isWasmDisabled) {
        try {
            wasmCheck = require('wasm-check');
            const tfliteTimeout = 10000;

            if (wasmCheck?.feature?.simd) {
                tflite = await timeout(tfliteTimeout, createTFLiteSIMDModule());
            } else {
                tflite = await timeout(tfliteTimeout, createTFLiteModule());
            }
        } catch (err) {
            isWasmDisabled = true;

            if (err?.message === '408') {
                console.error('Jitsi virtual background: Failed to download tflite model!');
            } else {
                console.error('Jitsi virtual background: Looks like WebAssembly is disabled or not supported on this browser', err);
            }
            
            return;
        }
    } else if (isWasmDisabled) {
        console.error('Jitsi virtual background: wasm is disabled!')
        return;
    }

    const modelBufferOffset = tflite._getModelBufferMemoryOffset();
    const modelResponse = await fetch(models.modelLandscape);

    if (!modelResponse.ok) {
        throw new Error('Failed to download tflite model!');
    }

    const model = await modelResponse.arrayBuffer();

    tflite.HEAPU8.set(new Uint8Array(model), modelBufferOffset);

    tflite._loadModel(model.byteLength);

    const options = {
        ...segmentationDimensions.modelLandscape,
        virtualBackground
    };

    return new JitsiStreamBackgroundEffect(tflite, options);
}

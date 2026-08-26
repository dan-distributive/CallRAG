// Minimal AbortController/AbortSignal polyfill. DCP's own sandbox fetch
// implementation (fetch-factory.js) constructs a `Request` that uses
// `new AbortController()` internally, but that global isn't defined in this
// worker context -- crashes with "AbortController is not a constructor"
// before any of our own fetch calls even happen. Just enough shape for
// Request/fetch's own usage (a `.signal` with `.aborted` and basic
// listener support) -- not a full spec-compliant implementation.
if (typeof globalThis.AbortController === 'undefined') {
  class AbortSignal {
    constructor() {
      this.aborted = false;
      this.reason = undefined;
      this._listeners = [];
    }
    addEventListener(type, cb) {
      if (type === 'abort') this._listeners.push(cb);
    }
    removeEventListener(type, cb) {
      this._listeners = this._listeners.filter((l) => l !== cb);
    }
    dispatchEvent() {}
  }

  class AbortController {
    constructor() {
      this.signal = new AbortSignal();
    }
    abort(reason) {
      if (this.signal.aborted) return;
      this.signal.aborted = true;
      this.signal.reason = reason;
      this.signal._listeners.forEach((cb) => {
        try {
          cb();
        } catch (e) {
          /* ignore listener errors */
        }
      });
    }
  }

  globalThis.AbortController = AbortController;
  globalThis.AbortSignal = AbortSignal;
}

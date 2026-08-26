const vowels = require('./vowels');

module.exports = function charInfo(char) {
  if (!/[a-z]/i.test(char)) return 'punctuation';
  return vowels.has(char.toLowerCase()) ? 'vowel' : 'consonant';
};

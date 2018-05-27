import {Pipe, PipeTransform} from '@angular/core';

/**
 * Create the ability to loop over the received classification object inside a ngFor directive.
 * If used in combination with the sort
 */
@Pipe({name: 'iterateGenres'})
export class IterateGenresPipe implements PipeTransform {
  transform(value: Object, args?: string|any[]): Object[] {
    let retObj = Object.entries(value)
      .map(genre => ({ name: genre[0], value: genre[1] }));

    if (args === 'order-descending') {
      retObj = retObj.sort((g1, g2) => g2.value - g1.value);
    }
    else if (args === 'order-ascending') {
      retObj = retObj.sort((g1, g2) => g1.value - g2.value);        
    }

    return retObj;
  }
}